import io
import re
import zipfile
import gzip
import pandas as pd
import numpy as np

class DataPreprocessor:
    def process(self, raw_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
        report = {
            "original_format": "unknown",
            "original_rows": 0,
            "final_rows": 0,
            "original_cols": 0,
            "final_cols": 0,
            "headers_added": False,
            "whitespace_cleaned": False,
            "missing_values_standardized": 0,
            "duplicates_removed": 0,
            "dtype_conversions": {},
            "columns_renamed": {},
            "column_mapping": {},
            "normalized_to_original": {},
            "original_columns": [],
            "dropped_column_values": {},
            "uci_adult_detected": False,
            "warnings": []
        }

        # Step 1: Format Detection & Loading
        df, original_format, extra_meta, load_warnings = self._load_data(raw_bytes, filename)
        report["original_format"] = original_format
        report.update(extra_meta)
        report["warnings"].extend(load_warnings)
        
        # New Step: Empty Columns
        df, empty_cols = self._drop_empty_columns(df)
        if empty_cols:
            report["empty_cols_dropped"] = empty_cols

        report["original_rows"] = len(df)
        report["original_cols"] = len(df.columns)

        # Step 2: Header Detection
        df, headers_added = self._handle_headers(df)
        report["headers_added"] = headers_added

        # Step 3: Whitespace Cleaning
        df, whitespace_cleaned = self._clean_whitespace(df)
        report["whitespace_cleaned"] = whitespace_cleaned

        # Step 4: Missing Value Standardization
        df, missing_count = self._standardize_missing_values(df)
        report["missing_values_standardized"] = int(missing_count)

        # Step 5: Duplicate Row Removal
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        report["duplicates_removed"] = int(initial_rows - len(df))

        # Step 6: Dtype Inference
        df, dtype_conversions = self._infer_dtypes(df)
        report["dtype_conversions"] = dtype_conversions

        # Step 7: Column Name Normalization
        df, renamed, column_mapping, normalized_to_orig = self._normalize_columns(df)
        report["columns_renamed"] = renamed
        report["column_mapping"] = column_mapping
        report["normalized_to_original"] = normalized_to_orig
        report["original_columns"] = list(column_mapping.keys())

        # Step 8: UCI Adult Specific Fix
        df, uci_detected = self._uci_adult_fix(df, filename)
        report["uci_adult_detected"] = uci_detected
        if uci_detected:
            report["detected_scenario"] = "income"

        # New Step: ID Columns
        id_cols = self._detect_id_columns(df)
        if id_cols:
            report["id_columns_detected"] = id_cols
            for col in id_cols:
                report["warnings"].append(f"Column '{col}' looks like a row ID. Consider excluding it from analysis.")

        # New Step: Zero Variance Columns
        df, zero_var_cols, dropped_values = self._drop_zero_variance_columns(df)
        if zero_var_cols:
            report["zero_variance_cols_dropped"] = zero_var_cols
            report["dropped_column_values"] = dropped_values

        report["final_rows"] = len(df)
        report["final_cols"] = len(df.columns)

        return df, report

    def _load_data(self, raw_bytes: bytes, filename: str) -> tuple[pd.DataFrame, str, dict, list]:
        ext = filename.lower().split('.')[-1]
        extra = {}
        warnings = []
        
        # Handle compressed files first
        if ext == 'zip':
            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                    file_list = z.namelist()
                    if not file_list:
                        raise ValueError("ZIP archive is empty.")
                    
                    files_info = [info for info in z.infolist() if not info.is_dir()]
                    if not files_info:
                        raise ValueError("ZIP archive contains no files.")
                    
                    files_info.sort(key=lambda x: x.file_size, reverse=True)
                    
                    target_info = None
                    for info in files_info:
                        lower_name = info.filename.lower()
                        if any(lower_name.endswith(e) for e in ['.csv', '.tsv', '.data', '.txt']):
                            if "readme" not in lower_name and "description" not in lower_name:
                                target_info = info
                                break
                    
                    if target_info is None:
                        target_info = files_info[0]
                        
                    raw_bytes = z.read(target_info.filename)
                    filename = target_info.filename
                    ext = filename.lower().split('.')[-1]
                    warnings.append(f"Extracted '{filename}' from ZIP archive.")
            except Exception as e:
                raise ValueError(f"Could not extract ZIP file: {str(e)}")
                
        elif ext == 'gz':
            try:
                raw_bytes = gzip.decompress(raw_bytes)
                ext = 'csv'  # Treat decompressed contents as CSV by default
            except Exception as e:
                raise ValueError(f"Could not decompress GZ file: {str(e)}")

        df = None
        fmt = ext

        try:
            if ext == 'csv':
                try:
                    df = pd.read_csv(io.BytesIO(raw_bytes))
                except Exception:
                    df = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine='python')
            elif ext == 'tsv':
                df = pd.read_csv(io.BytesIO(raw_bytes), sep='\t')
            elif ext in ['xlsx', 'xls']:
                engine = 'openpyxl' if ext == 'xlsx' else 'xlrd'
                xls = pd.ExcelFile(io.BytesIO(raw_bytes), engine=engine)
                sheet_names = xls.sheet_names
                sheet = sheet_names[0]
                df = pd.read_excel(xls, sheet_name=sheet)
                extra["excel_sheet_used"] = sheet
                if len(sheet_names) > 1:
                    warnings.append(f"Excel file has {len(sheet_names)} sheets. Using '{sheet}'. Upload a specific sheet if needed.")
            elif ext == 'json':
                try:
                    df = pd.read_json(io.BytesIO(raw_bytes), orient='records')
                except Exception:
                    df = pd.read_json(io.BytesIO(raw_bytes), orient='columns')
            elif ext == 'parquet':
                df = pd.read_parquet(io.BytesIO(raw_bytes))
            elif ext in ['data', 'txt']:
                try:
                    df = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine='python')
                except Exception:
                    df = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine='python', header=None)
            else:
                # Fallback: try reading as CSV
                try:
                    df = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine='python')
                    fmt = "csv (inferred)"
                except Exception:
                    raise ValueError(f"Unsupported or unparseable file format: {ext}")
                    
        except Exception as e:
            raise ValueError(f"Could not read file as {ext}. Please ensure it is a valid tabular file. Error: {str(e)}")
            
        if df is None or df.empty:
            raise ValueError("The parsed file contains no data.")
            
        return df, fmt, extra, warnings

    def _handle_headers(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        # Check if all column names are integers or match Unnamed pattern
        cols = list(df.columns)
        all_numeric_or_unnamed = all(
            str(c).isdigit() or str(c).startswith("Unnamed:") 
            for c in cols
        )
        
        if all_numeric_or_unnamed:
            df.columns = [f"col_{i}" for i in range(len(cols))]
            return df, True
            
        return df, False

    def _clean_whitespace(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        cleaned = False
        # Strip whitespace from column names
        original_cols = list(df.columns)
        df.columns = [str(c).strip() for c in df.columns]
        if list(df.columns) != original_cols:
            cleaned = True

        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object', 'string']).columns:
            # Using str.strip() and replacing empty strings with NaN
            s = df[col].astype(str).str.strip()
            # Restore original non-string NaNs if any
            mask = df[col].isna()
            s = s.replace('', np.nan)
            s[mask] = np.nan
            
            # Check if anything changed
            if not df[col].equals(s):
                df[col] = s
                cleaned = True
                
        return df, cleaned

    def _standardize_missing_values(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        missing_indicators = [
            '?', 'N/A', 'NA', 'n/a', 'na', 'null', 'NULL', 'None', '-', '--', 'missing', 'MISSING'
        ]
        
        initial_nas = df.isna().sum().sum()
        df.replace(missing_indicators, np.nan, inplace=True)
        final_nas = df.isna().sum().sum()
        
        return df, final_nas - initial_nas

    def _infer_dtypes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        conversions = {}
        for col in df.select_dtypes(include=['object', 'string']).columns:
            series = df[col]
            # Ignore completely empty columns
            if series.isna().all():
                continue
                
            numeric_series = pd.to_numeric(series, errors='coerce')
            non_null_original = series.notna().sum()
            non_null_converted = numeric_series.notna().sum()
            
            if non_null_original > 0 and (non_null_converted / non_null_original) >= 0.8:
                df[col] = numeric_series
                conversions[col] = "converted to numeric"
                
        return df, conversions

    def _normalize_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, dict]:
        renamed = {}
        column_mapping = {}
        normalized_to_original = {}
        new_cols = []
        original_cols = list(df.columns)
        
        for col in original_cols:
            new_col = str(col).lower()
            new_col = re.sub(r'[\s\-]+', '_', new_col)
            new_col = re.sub(r'[^a-z0-9_]', '', new_col)
            
            # Detect collisions where distinct original columns map to the same normalized name
            if new_col in normalized_to_original and normalized_to_original[new_col] != col:
                raise ValueError(
                    f"Column name ambiguity detected: columns '{normalized_to_original[new_col]}' "
                    f"and '{col}' both normalize to '{new_col}'."
                )
                
            column_mapping[col] = new_col
            normalized_to_original[new_col] = col
            if new_col != col:
                renamed[col] = new_col
            new_cols.append(new_col)
            
        df.columns = new_cols
        return df, renamed, column_mapping, normalized_to_original

    def _uci_adult_fix(self, df: pd.DataFrame, filename: str) -> tuple[pd.DataFrame, bool]:
        adult_keywords = ['workclass', 'fnlwgt', 'education_num', 'marital_status', 'occupation']
        
        is_adult_by_name = 'adult' in filename.lower()
        is_adult_by_cols = any(kw in df.columns for kw in adult_keywords)
        
        if not (is_adult_by_name or is_adult_by_cols):
            return df, False
            
        expected_cols = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num', 
            'marital_status', 'occupation', 'relationship', 'race', 'sex', 
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
        ]
        
        if len(df.columns) == len(expected_cols):
            df.columns = expected_cols
            
        if 'income' in df.columns:
            income_col = df['income'].astype(str).str.strip()
            df['income_binary'] = np.where(income_col == '>50K', 1, np.where(income_col == '<=50K', 0, np.nan))
            
        return df, True

    def _drop_empty_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        empty_cols = []
        for col in df.columns:
            if df[col].isna().mean() > 0.95:
                empty_cols.append(col)
        if empty_cols:
            df.drop(columns=empty_cols, inplace=True)
        return df, empty_cols

    def _detect_id_columns(self, df: pd.DataFrame) -> list:
        id_cols = []
        for col in df.columns:
            lower_col = str(col).lower()
            if "id" in lower_col or "index" in lower_col:
                if df[col].nunique(dropna=False) == len(df):
                    id_cols.append(col)
        return id_cols

    def _drop_zero_variance_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list, dict]:
        zero_var_cols = []
        dropped_values = {}
        for col in df.columns:
            if df[col].nunique(dropna=False) <= 1:
                zero_var_cols.append(col)
                val = df[col].dropna().iloc[0] if df[col].notna().any() else 0
                dropped_values[col] = val
        if zero_var_cols:
            df.drop(columns=zero_var_cols, inplace=True)
        return df, zero_var_cols, dropped_values
