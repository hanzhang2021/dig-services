from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import os
import pronto
import requests
from functools import lru_cache
from scipy.stats import t as student_t

app = Flask(__name__)

# --- CONFIGURATION ---
PORT = 4999
DATA_FILE = 'sample-hpo-Wp-crdc2.tsv'
ID_COLUMN = 'unique_sample_ID'
SAMPLE_ID_COLUMN = 'SampleInVCF'
ONTOLOGY_FILE = 'hp.obo'
COVARIATE_FILE = 'CRDC15761_covariates_20261019.tsv'
GENE_VARIANTS_CRDC_URL = 'http://100.80.30.199:5000/api/bio/query/gene-variants-crdc'
GENE_VARIANTS_CRDC_CONT_URL = 'http://100.80.30.199:5000/api/bio/cont'

# Global variables for fast access
NEW_DF = None
HPO_COLS = []
HPO_DICT = {}
# Per-sample overall phenotyping intensity: total number of HPO terms
# present (matches rphers_fast.py's `total_hpo_terms`, i.e. the row sums of
# the BINARY presence matrix, not a Wp-weighted sum). Precomputed once at
# startup and reused as the regression covariate for Residual PheRS.
TOTAL_HPO_TERM_COUNT = None
HPO_ROW_INDEX = {}
COVARIATE_SAMPLE_IDS = []
COVARIATE_VALUES = None
COVARIATE_NAMES = []
COVARIATE_INDEX = {}
COVARIATE_Q = None
COVARIATE_DF = None
VARIANT_ASSOCIATION_MODEL = 'portal_variant_binary_ols_covariate_v1'


def _canonical_sex(value):
    value = str(value or '').strip().lower()
    if value in {'female', 'f'}:
        return 'Female'
    if value in {'male', 'm'}:
        return 'Male'
    return 'Unknown'


def independent_covariate_design(covariates, names):
    """Keep the intercept and non-aliased covariates in their original order.

    Identical to scripts/variant_association.py::independent_covariate_design.
    """
    n = covariates.shape[0]
    columns = [np.ones(n)]
    q = np.ones((n, 1)) / np.sqrt(n)
    dropped = []
    for name, column in zip(names, covariates.T):
        residual = column - q @ (q.T @ column)
        residual -= q @ (q.T @ residual)
        if np.linalg.norm(residual) <= 1e-10 * max(1.0, np.linalg.norm(column)):
            dropped.append(name)
            continue
        columns.append(column)
        q = np.column_stack((q, residual / np.linalg.norm(residual)))
    return np.column_stack(columns), tuple(dropped)


def load_covariates():
    """Load the CRDC covariates in the same encoding as variant_association.py."""
    global COVARIATE_SAMPLE_IDS, COVARIATE_VALUES, COVARIATE_NAMES
    global COVARIATE_INDEX, COVARIATE_Q, COVARIATE_DF

    if NEW_DF is None or not os.path.exists(COVARIATE_FILE):
        print(f"Warning: {COVARIATE_FILE} not found; variant effects unavailable.")
        return
    covariates = pd.read_csv(COVARIATE_FILE, sep='\t')
    required = {'sample_id', 'vcf_sample_id', 'age', 'sex', *[f'PC{i}' for i in range(1, 11)]}
    missing = required.difference(covariates.columns)
    if missing:
        raise ValueError(f"Covariate file is missing columns: {', '.join(sorted(missing))}")

    # BioIndex variant rows use the VCF/sample identifier, not the browser's
    # unique_sample_ID.  Keep the analysis roster in that same identifier
    # space so carrier membership and phenotype rows line up exactly.
    hpo_ids = set(NEW_DF[SAMPLE_ID_COLUMN].astype(str))
    covariates['vcf_sample_id'] = covariates['vcf_sample_id'].astype(str).str.strip()
    covariates = covariates[covariates['vcf_sample_id'].isin(hpo_ids)].copy()
    covariates = covariates.drop_duplicates('vcf_sample_id', keep=False)
    if covariates.empty:
        print('Warning: no HPO/covariate sample overlap; variant effects unavailable.')
        return

    ages = pd.to_numeric(covariates['age'], errors='coerce')
    valid_ages = ages.where(ages.between(0, 99))
    age_median = float(valid_ages.median())
    if not np.isfinite(age_median):
        raise ValueError('Covariate file has no observed age values in the analysis roster')
    age_missing = valid_ages.isna().astype(float)
    age_values = valid_ages.fillna(age_median)
    sex = covariates['sex'].map(_canonical_sex)
    pc_names = [f'PC{i}' for i in range(1, 11)]
    COVARIATE_NAMES = ['age', 'age_missing', 'sex_male', 'sex_unknown', *pc_names]
    values = np.column_stack([
        age_values.to_numpy(float), age_missing.to_numpy(float),
        (sex == 'Male').to_numpy(float), (sex == 'Unknown').to_numpy(float),
        covariates[pc_names].apply(pd.to_numeric, errors='raise').to_numpy(float),
    ])
    COVARIATE_SAMPLE_IDS = covariates['vcf_sample_id'].tolist()
    COVARIATE_VALUES = values
    # A variant's `samples` array is keyed by vcf_sample_id only, so each
    # carrier maps to exactly one covariate row (no double-counting).
    COVARIATE_INDEX = {}
    for index, vcf_id in enumerate(covariates['vcf_sample_id']):
        if vcf_id:
            if vcf_id in COVARIATE_INDEX:
                raise ValueError(f"duplicate vcf_sample_id in covariate roster: {vcf_id}")
            COVARIATE_INDEX[vcf_id] = index

    # Match independent_covariate_design(): intercept plus non-aliased columns.
    design, dropped = independent_covariate_design(values, COVARIATE_NAMES)
    if len(values) <= design.shape[1] + 1:
        raise ValueError('Covariate design has no residual degrees of freedom')
    q, _ = np.linalg.qr(design, mode='reduced')
    COVARIATE_Q = q
    COVARIATE_DF = len(values) - design.shape[1] - 1
    if dropped:
        print(f"Dropped aliased covariates: {', '.join(dropped)}")
    print(f"Covariates Loaded: {len(values)} samples, {len(COVARIATE_NAMES)} columns "
          f"({design.shape[1]} design columns incl. intercept).")

def load_data_and_ontology():
    global NEW_DF, HPO_COLS, HPO_DICT, TOTAL_HPO_TERM_COUNT, HPO_ROW_INDEX

    # 1. Load your Matrix Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please ensure the weighted TSV is present.")
        return
    try:
        # The 949 MB matrix expands beyond available RAM when pandas parses
        # every phenotype value as float64. HPO weights do not need float64
        # precision, so keep them in float32 and avoid a full-width temporary
        # boolean frame during startup.
        header = pd.read_csv(DATA_FILE, sep='\t', nrows=0)
        HPO_COLS = [col for col in header.columns if col.startswith('HP:')]
        NEW_DF = pd.read_csv(
            DATA_FILE,
            sep='\t',
            dtype={column: np.float32 for column in HPO_COLS},
            low_memory=False,
        )
        HPO_ROW_INDEX = {}
        for index, vcf_id in enumerate(NEW_DF[SAMPLE_ID_COLUMN].astype(str)):
            vcf_id = vcf_id.strip()
            if vcf_id:
                HPO_ROW_INDEX[vcf_id] = index
        # NEW_DF already stores Wp * presence-indicator per cell (i.e. 0 when
        # absent, the term's Wp weight when present), so presence is simply
        # "value > 0" -- no need to re-derive Wp from prevalence counts.
        TOTAL_HPO_TERM_COUNT = np.zeros(len(NEW_DF), dtype=np.int32)
        for start in range(0, len(HPO_COLS), 256):
            block = NEW_DF[HPO_COLS[start:start + 256]].to_numpy(copy=False)
            TOTAL_HPO_TERM_COUNT += np.count_nonzero(block > 0, axis=1)
        print(f"Matrix Loaded: {len(NEW_DF)} samples and {len(HPO_COLS)} terms.")
    except Exception as e:
        print(f"Failed to load data matrix: {e}")

    # 2. Load the Ontology Names Dictionary
    if not os.path.exists(ONTOLOGY_FILE):
        print(f"Warning: {ONTOLOGY_FILE} not found. Phenotype names will show as 'Unknown'.")
        return
    try:
        print("Parsing HPO Ontology file...")
        ont = pronto.Ontology(ONTOLOGY_FILE)
        # Map term IDs to their text names safely
        HPO_DICT = {term.id: term.name for term in ont.terms() if term.name}
        print(f"Ontology Loaded: Successfully cached names for {len(HPO_DICT)} terms.")
    except Exception as e:
        print(f"Failed to parse ontology file: {e}")

    try:
        load_covariates()
    except Exception as e:
        print(f"Failed to load covariates: {e}")

def get_phenotype_name(hpo_id):
    """Safely looks up text name or falls back to 'Unknown'"""
    return HPO_DICT.get(hpo_id, "Unknown Phenotype Name")


@lru_cache(maxsize=32)
def residual_score_vector(hpo_terms):
    """Compute residual PheRS once per HPO query, not once per variant."""
    if NEW_DF is None or TOTAL_HPO_TERM_COUNT is None:
        return None
    terms = tuple(term for term in hpo_terms if term in HPO_COLS)
    if not terms:
        return None
    raw_all = NEW_DF[list(terms)].sum(axis=1).to_numpy(float)
    intensity = np.asarray(TOTAL_HPO_TERM_COUNT, dtype=float)
    intensity_design = np.column_stack((np.ones(len(NEW_DF)), intensity))
    intensity_fit, *_ = np.linalg.lstsq(intensity_design, raw_all, rcond=None)
    return raw_all - intensity_design @ intensity_fit


@lru_cache(maxsize=32)
def variant_outcome(hpo_terms):
    """Prepare the covariate-residualized phenotype vector once per query."""
    if COVARIATE_Q is None or not hpo_terms:
        return None
    residual_all = residual_score_vector(hpo_terms)
    if residual_all is None:
        return None
    analysis_indices = [HPO_ROW_INDEX[sample] for sample in COVARIATE_SAMPLE_IDS if sample in HPO_ROW_INDEX]
    if len(analysis_indices) != len(COVARIATE_SAMPLE_IDS):
        return None
    y = residual_all[analysis_indices]
    y = y - COVARIATE_Q @ (COVARIATE_Q.T @ y)
    return y


def calculate_variant_association(samples, hpo_terms_list):
    """Calculate the portal's covariate-adjusted binary variant association."""
    unavailable = {
        'beta': None, 'p_value': None, 'status': 'unavailable',
        'n_samples': len(COVARIATE_SAMPLE_IDS),
        'n_carriers': None,
        'model_version': VARIANT_ASSOCIATION_MODEL,
    }
    if COVARIATE_Q is None or COVARIATE_VALUES is None or not hpo_terms_list:
        return unavailable

    terms = tuple(sorted({term for term in hpo_terms_list if term in HPO_COLS}))
    if not terms:
        return {
            **unavailable,
            'status': 'missing_hpo_terms',
            'missing_hpo_terms': list(dict.fromkeys(hpo_terms_list)),
        }
    y = variant_outcome(terms)
    if y is None:
        return unavailable
    yy = float(y @ y)
    carrier_ids = {str(sample).strip() for sample in (samples or [])}
    carrier_ids.discard('')
    # variant_association.py raises when a carrier is outside the analysis
    # roster (never silently drops it). Per-record, report the same condition
    # as a status instead of failing the whole response.
    unknown = [sample for sample in carrier_ids if sample not in COVARIATE_INDEX]
    if unknown:
        return {**unavailable, 'status': 'carriers_outside_analysis_roster',
                'n_carriers': len(carrier_ids), 'n_carriers_outside_roster': len(unknown)}
    carrier_indices = np.asarray(
        [COVARIATE_INDEX[sample] for sample in carrier_ids],
        dtype=np.int64,
    )
    count = len(carrier_indices)
    if count == 0:
        return {**unavailable, 'status': 'no_carriers', 'n_carriers': 0, 'low_carrier_count': True}
    if count == len(COVARIATE_SAMPLE_IDS):
        return {**unavailable, 'status': 'all_carriers', 'n_carriers': count, 'low_carrier_count': False}
    if yy <= 1e-12:
        return {**unavailable, 'status': 'constant_outcome', 'n_carriers': count, 'low_carrier_count': count < 10}

    qx = COVARIATE_Q[carrier_indices].sum(axis=0)
    xx = float(count - qx @ qx)
    if xx <= 1e-10 or COVARIATE_DF <= 0:
        return {**unavailable, 'status': 'constant_or_collinear_predictor', 'n_carriers': count,
                'low_carrier_count': count < 10}
    xy = float(y[carrier_indices].sum())
    beta = xy / xx
    rss = max(0.0, yy - xy * xy / xx)
    if rss <= 1e-12:
        return {'beta': beta, 'p_value': None, 'status': 'zero_residual_variance',
                'n_samples': len(COVARIATE_SAMPLE_IDS), 'n_carriers': count,
                'low_carrier_count': count < 10, 'model_version': VARIANT_ASSOCIATION_MODEL}
    standard_error = float(np.sqrt(rss / COVARIATE_DF / xx))
    p_value = float(2 * student_t.sf(abs(beta / standard_error), COVARIATE_DF))
    return {
        'beta': float(beta), 'p_value': p_value, 'standard_error': standard_error,
        'status': 'ok', 'n_samples': len(COVARIATE_SAMPLE_IDS), 'n_carriers': count,
        'low_carrier_count': count < 10, 'model_version': VARIANT_ASSOCIATION_MODEL,
    }

# Initialize data engines on startup
load_data_and_ontology()


@app.route('/analyze', methods=['POST'])
def analyze_symptoms():
    """
    Endpoint 1: Find similar samples based on input HPO terms.
    """
    if NEW_DF is None:
        return jsonify({"error": "Data not loaded on server"}), 500

    user_entered_terms = []
    if request.is_json:
        data = request.get_json()
        if data and 'terms' in data:
            if isinstance(data['terms'], list):
                user_entered_terms = data['terms']
            elif isinstance(data['terms'], str):
                user_entered_terms = data['terms'].split(',')

    if not user_entered_terms:
        terms_param = request.args.get('terms') or request.args.get('query')
        if terms_param:
            user_entered_terms = terms_param.split(',')

    user_entered_terms = [str(t).strip() for t in user_entered_terms if str(t).strip()]
    valid_user_terms = [t for t in user_entered_terms if t in HPO_COLS]
    total_valid_input_count = len(valid_user_terms)

    try:
        # Only touch the queried columns: copying NEW_DF and dotting every HPO
        # column allocates ~2 GB per request. The score is the sum of Wp weights
        # across the valid query terms (same as the previous dot product with a
        # 0/1 indicator vector).
        non_hpo_cols = [c for c in NEW_DF.columns if not c.startswith('HP:')]
        if valid_user_terms:
            similarity = NEW_DF[valid_user_terms].sum(axis=1)
            matches = (NEW_DF[valid_user_terms] > 0).sum(axis=1).astype(str) + f"/{total_valid_input_count}"
        else:
            similarity = pd.Series(0.0, index=NEW_DF.index)
            matches = pd.Series(f"0/{total_valid_input_count}", index=NEW_DF.index)

        top_index = similarity.sort_values(ascending=False).head(30).index
        ranked = NEW_DF.loc[top_index, non_hpo_cols].copy()
        ranked['similarity_score'] = similarity.loc[top_index]
        ranked['matches'] = matches.loc[top_index]
        output = ranked.to_dict(orient='records')

        return jsonify({
            "query_received": ", ".join(user_entered_terms),
            "total_query_terms_submitted": len(user_entered_terms),
            "valid_query_terms_used_as_denominator": total_valid_input_count,
            "top_matches": output
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/top_phenotypes', methods=['POST'])
def top_phenotypes():
    """
    Endpoint 2: Take a list of samples, return top 20 highest-occurring HPO terms.
    Now includes name tracking mapping.
    """
    if NEW_DF is None:
        return jsonify({"error": "Data not loaded on server"}), 500

    data = request.get_json()
    if not data or 'samples' not in data:
        return jsonify({"error": "Please provide a JSON body containing a 'samples' array."}), 400

    requested_samples = [str(s).strip() for s in data['samples'] if str(s).strip()]

    if ID_COLUMN not in NEW_DF.columns:
        return jsonify({"error": f"ID column '{ID_COLUMN}' not found in the loaded dataset."}), 500

    try:
        filtered_df = NEW_DF[NEW_DF[ID_COLUMN].astype(str).isin(requested_samples)]
        total_samples_found = len(filtered_df)

        if total_samples_found == 0:
            return jsonify({
                "samples_submitted": len(requested_samples),
                "samples_found": 0,
                "top_phenotypes": []
            }), 200

        counts = (filtered_df[HPO_COLS] > 0).sum()
        wps = filtered_df[HPO_COLS].max()

        summary_df = pd.DataFrame({'count': counts, 'Wp': wps})
        summary_df = summary_df[summary_df['count'] > 0]

        sorted_summary = summary_df.sort_values(by=['count', 'Wp'], ascending=[False, False]).head(20)
        
        output = []
        for hpo, row in sorted_summary.iterrows():
            phenotype_count = int(row['count'])
            hpo_id_str = str(hpo)
            output.append({
                "hpo_id": hpo_id_str,
                "name": get_phenotype_name(hpo_id_str),  # Updated inclusion mapping
                "count": phenotype_count,
                "Wp": float(row['Wp']),
                "matches": f"{phenotype_count}/{total_samples_found}"
            })

        return jsonify({
            "samples_submitted": len(requested_samples),
            "samples_found": total_samples_found,
            "top_phenotypes": output
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/sample_phenotypes', methods=['POST'])
def sample_phenotypes():
    """
    Endpoint 3: Take one sample ID, return all its present phenotypes ordered by Wp.
    Now includes name tracking mapping.
    """
    if NEW_DF is None:
        return jsonify({"error": "Data not loaded on server"}), 500

    data = request.get_json()
    if not data or 'sample_id' not in data:
        return jsonify({"error": "Please provide a JSON body containing a 'sample_id' string."}), 400

    target_sample = str(data['sample_id']).strip()

    if ID_COLUMN not in NEW_DF.columns:
        return jsonify({"error": f"ID column '{ID_COLUMN}' not found in the loaded dataset."}), 500

    try:
        sample_row = NEW_DF[NEW_DF[ID_COLUMN].astype(str) == target_sample]

        if sample_row.empty:
            return jsonify({"error": f"Sample ID '{target_sample}' not found in the database."}), 404

        hpo_series = sample_row[HPO_COLS].squeeze()
        if isinstance(hpo_series, pd.DataFrame):
            hpo_series = hpo_series.iloc[0]

        present_phenotypes = hpo_series[hpo_series > 0]
        sorted_phenotypes = present_phenotypes.sort_values(ascending=False)

        output = []
        for hpo, wp in sorted_phenotypes.items():
            hpo_id_str = str(hpo)
            output.append({
                "hpo_id": hpo_id_str,
                "name": get_phenotype_name(hpo_id_str),  # Updated inclusion mapping
                "Wp": float(wp)
            })

        return jsonify({
            "sample_id": target_sample,
            "total_phenotypes": len(output),
            "phenotypes": output
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def calculate_match_score(samples, hpo_terms_list):
    """
    Residual PheRS, following scripts/rphers_fast.py's score_samples as the
    reference implementation:

      raw_score_i = sum over the queried HPO terms (valid_terms) of NEW_DF's
                    per-cell value for sample i. NEW_DF already stores
                    Wp * presence-indicator per cell (0 when absent, the
                    term's Wp weight when present), so this sum is exactly
                    the reference's `matched_matrix @ weights[matched_index]`
                    -- no need to re-derive Wp from prevalence counts.

      total_hpo_i = TOTAL_HPO_TERM_COUNT[i], the total number of HPO terms
                    present for sample i (reference's `total_hpo_terms`),
                    used as the regression covariate for phenotyping
                    intensity.

      residual_i  = raw_score_i - fitted(raw_score ~ 1 + total_hpo), with the
                    regression fit via least squares (np.linalg.lstsq) across
                    the FULL population, exactly as rphers_fast.py does
                    (design = [ones, total_hpo_terms]).

    Returns the mean residual across the record's matched samples.
    """
    if NEW_DF is None or TOTAL_HPO_TERM_COUNT is None or SAMPLE_ID_COLUMN not in NEW_DF.columns:
        return 0.0

    sample_ids = [str(s).strip() for s in (samples or []) if str(s).strip()]
    valid_terms = tuple(sorted({t for t in (hpo_terms_list or []) if t in HPO_COLS}))

    if not sample_ids or not valid_terms:
        return 0.0

    sample_indices = [HPO_ROW_INDEX[sample_id] for sample_id in sample_ids if sample_id in HPO_ROW_INDEX]
    if not sample_indices:
        return 0.0

    residual_all = residual_score_vector(valid_terms)
    if residual_all is None:
        return 0.0
    return float(np.mean(residual_all[sample_indices]))


def annotate_gene_variant_records(result, hpo_terms_list=None):
    """
    Walks the 'data' section of the gene-variants-crdc response and adds a
    'match_score' field to each record:
      - "unavailable" when the record's samples array is empty
      - the calculated Wp-based score (as a string) otherwise
    """
    # Accepts either the response dict (with a 'data' list) or the list itself.
    if isinstance(result, list):
        records = result
    elif isinstance(result, dict):
        records = result.get('data')
    else:
        return result

    if not isinstance(records, list):
        return result

    for record in records:
        if not isinstance(record, dict):
            continue

        samples = record.get('samples') or []
        #hpo_terms_list = record.get('hpo_terms_list') or []

        if not samples:
            record['match_score'] = "unavailable"
        else:
            score = calculate_match_score(samples, hpo_terms_list)
            record['match_score'] = str(score)

        association = calculate_variant_association(samples, hpo_terms_list)
        # Keep the descriptive API names alongside the algorithm's canonical
        # beta/p_value fields so callers can render the columns directly.
        record['effect_score_beta'] = association.get('beta')
        record['effect_score_p_value'] = association.get('p_value')
        record['variant_association'] = association

    return result


@app.route('/gene-variants-crdc', methods=['POST'])
def gene_variants_crdc():
    """
    Endpoint 4: Take a gene name and proxy the request to the
    gene-variants-crdc service running on port 5000, returning its result.
    """
    gene_value = None
    hpo_terms = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        gene_value = data.get('gene')
        print(gene_value)
        hpo_terms = data.get('hpo_terms')
        print(hpo_terms)

    if not gene_value:
        gene_value = request.args.get('gene')
    
    if not gene_value or not str(gene_value).strip():
        return jsonify({"error": "Please provide a 'gene' value."}), 400

    gene_value = str(gene_value).strip()

    if hpo_terms is None:
        hpo_terms = request.args.get('hpo_terms')

    # hpo_terms is optional (the gene view does not always have HPO terms to
    # send); a missing/empty value just means match scores stay "unavailable"
    # rather than a hard validation error.
    if hpo_terms is None:
        hpo_terms = []

    gene_value = str(gene_value).strip()

    # hpo_terms may already be a list (from JSON body) or a comma-separated
    # string (from query params / a JSON string value). Normalize to a list
    # without stringifying an actual list (which would turn it into
    # "['HP:...', ...]" and get iterated character-by-character downstream).
    if isinstance(hpo_terms, list):
        hpo_terms_list = [str(t).strip() for t in hpo_terms if str(t).strip()]
    else:
        hpo_terms_list = [t.strip() for t in str(hpo_terms).split(',') if t.strip()]

    # Normalize like variant_association.analyze_gene_variants: strip + uppercase,
    # drop empties and duplicates (applies to match score and association).
    hpo_terms_list = list(dict.fromkeys(t.strip().upper() for t in hpo_terms_list if t.strip()))
    print(hpo_terms_list)


    try:
        response = requests.get(GENE_VARIANTS_CRDC_URL, params={'q': gene_value}, timeout=30)
        response.raise_for_status()
        try:
            result = response.json()
        except ValueError:
            return jsonify(response.text)

        # Collect result.data, following result.continuation until exhausted.
        all_data = []
        if isinstance(result, dict):
            page = result
            while True:
                page_data = page.get('data')
                if isinstance(page_data, list):
                    all_data.extend(page_data)
                token = page.get('continuation')
                if not token:
                    break
                cont_response = requests.get(
                    GENE_VARIANTS_CRDC_CONT_URL,
                    params={'token': token},
                    timeout=30,
                )
                cont_response.raise_for_status()
                page = cont_response.json()
            result['data'] = all_data
            result.pop('continuation', None)

        annotate_gene_variant_records(all_data, hpo_terms_list)
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to reach gene-variants-crdc service: {e}"}), 502


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT)
