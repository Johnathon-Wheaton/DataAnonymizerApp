import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
import re
from io import BytesIO
from faker import Faker

# Initialize Faker for generating realistic fake data
fake = Faker()

# All supported data types — displayed in dropdowns for user override
ALL_DATA_TYPES = [
    'leave_unchanged',
    'text',
    'name',
    'email',
    'phone',
    'username',
    'job_title',
    'company',
    'address',
    'city',
    'state',
    'state_code',
    'country',
    'country_code',
    'postal_code',
    'latitude',
    'longitude',
    'integer',
    'float',
    'currency',
    'datetime',
    'date_of_birth',
    'boolean',
    'categorical',
    'id',
    'ssn',
    'credit_card',
    'ip_address',
    'url',
    'file_path',
    'license_plate',
    'unknown',
]

class DataAnonymizer:
    def __init__(self):
        self.fake = Faker()
        self.location_cache = {}
        
    def detect_data_type(self, series):
        """Detect the data type of a pandas series"""
        # Skip if all values are null
        if series.dropna().empty:
            return 'unknown'

        # Get non-null values for analysis
        non_null = series.dropna()
        sample_values = non_null.astype(str).str.strip()
        col_lower = series.name.lower() if series.name else ''

        # Check for boolean dtype
        if pd.api.types.is_bool_dtype(series):
            return 'boolean'

        # Check for native datetime dtype first (Excel dates parsed by pandas)
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'

        # Check for numeric types — must come before datetime since
        # pd.to_datetime aggressively parses plain numbers as dates.
        # Check lat/long by column name before returning generic numeric.
        if pd.api.types.is_numeric_dtype(series):
            if 'lat' in col_lower and non_null.between(-90, 90).all():
                return 'latitude'
            if ('lon' in col_lower or 'lng' in col_lower) and non_null.between(-180, 180).all():
                return 'longitude'
            if pd.api.types.is_integer_dtype(series):
                return 'integer'
            return 'float'

        # Try to convert string values to numeric (handles strings like "123")
        try:
            numeric_vals = pd.to_numeric(non_null)
            if 'lat' in col_lower and numeric_vals.between(-90, 90).all():
                return 'latitude'
            if ('lon' in col_lower or 'lng' in col_lower) and numeric_vals.between(-180, 180).all():
                return 'longitude'
            if all(numeric_vals % 1 == 0):
                return 'integer'
            return 'float'
        except (ValueError, TypeError):
            pass

        # Try to parse as datetime — only for string values with date-like structure
        if self._looks_like_date(sample_values.head(10)):
            try:
                pd.to_datetime(non_null.head(10))
                return 'datetime'
            except (ValueError, TypeError):
                pass

        # Pattern matching for specific types
        sample_str = sample_values.head(20)

        # Check for boolean-like string values (Yes/No, True/False, Y/N, etc.)
        unique_lower = set(non_null.astype(str).str.strip().str.lower().unique())
        boolean_sets = [
            {'yes', 'no'}, {'y', 'n'}, {'true', 'false'}, {'t', 'f'},
            {'1', '0'}, {'on', 'off'}, {'active', 'inactive'},
            {'enabled', 'disabled'},
        ]
        if any(unique_lower == bset or unique_lower <= bset for bset in boolean_sets):
            return 'boolean'

        # Email pattern
        if sample_str.str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$').sum() > len(sample_str) * 0.5:
            return 'email'

        # Phone number pattern — require visible formatting (spaces, dashes,
        # parens, or leading +). Bare digit strings are NOT phones.
        is_phone_col = any(term in col_lower for term in ['phone', 'tel', 'mobile', 'cell', 'fax'])
        phone_pattern = re.compile(
            r'^\+\d[\d\s\-]{7,14}$'                          # +1 555 123 4567 (must start with +)
            r'|^\(?\d{2,4}\)[\s\-]?\d{3,4}[\s\-]?\d{3,4}$'  # (555) 123-4567 or (555)1234567
            r'|^\d{2,4}[\s\-]\d{3,4}[\s\-]\d{3,4}$'         # 555-123-4567 or 555 123 4567 (separators required)
        )
        phone_matches = sample_str.apply(lambda x: bool(phone_pattern.match(str(x))))
        if phone_matches.sum() > len(sample_str) * 0.5:
            return 'phone'
        # Only use column name hint if values are at least all digits/punctuation
        if is_phone_col:
            digit_like = sample_str.str.match(r'^[\d\s\-\(\)\+\.]+$').sum() > len(sample_str) * 0.5
            if digit_like:
                return 'phone'

        # Currency pattern (e.g. $1,234.56, €50.00, £1,000)
        currency_pattern = re.compile(r'^[\$€£¥₹]\s?[\d,]+\.?\d*$|^[\d,]+\.?\d*\s?[\$€£¥₹]$')
        if sample_str.apply(lambda x: bool(currency_pattern.match(str(x)))).sum() > len(sample_str) * 0.5:
            return 'currency'

        # Check column name for hints
        if any(term in col_lower for term in ['name', 'first', 'last', 'fname', 'lname']):
            return 'name'
        elif any(term in col_lower for term in ['address', 'street', 'addr']):
            return 'address'
        elif any(term in col_lower for term in ['city', 'town']):
            return 'city'
        elif any(term in col_lower for term in ['state', 'province']):
            if sample_str.str.len().median() <= 3:
                return 'state_code'
            return 'state'
        elif any(term in col_lower for term in ['country', 'nation']):
            if sample_str.str.len().median() <= 3:
                return 'country_code'
            return 'country'
        elif any(term in col_lower for term in ['zip', 'postal', 'postcode']):
            return 'postal_code'
        elif any(term in col_lower for term in ['company', 'organization', 'org']):
            return 'company'
        elif any(term in col_lower for term in ['desc', 'comment', 'note', 'remark', 'message', 'text', 'summary']):
            return 'text'
        elif any(term in col_lower for term in ['id', 'code', 'sku', 'ref', 'number', 'num', 'no']):
            return 'id'
        elif any(term in col_lower for term in ['email', 'e-mail']):
            return 'email'

        # Check for common patterns in data
        unique_ratio = len(non_null.unique()) / len(non_null) if len(non_null) > 0 else 0
        n_unique = len(non_null.unique())

        # Low cardinality (few unique values) — likely a categorical column, preserve by shuffling
        if n_unique <= 10 and len(non_null) > 10:
            return 'categorical'

        # High cardinality text — use column name + structure to decide
        if unique_ratio > 0.8:
            avg_words = sample_str.str.split().str.len().mean()
            avg_len = sample_str.str.len().mean()
            # Multi-word, moderate length → likely names (not long descriptions)
            if 2 <= avg_words <= 4 and avg_len < 40:
                return 'name'
            # Long multi-word text → descriptions/comments
            elif avg_words > 4 or avg_len >= 40:
                return 'text'
            else:
                return 'id'

        # Check for city/state/country patterns
        if self._check_location_pattern(sample_str, 'city'):
            return 'city'
        elif self._check_location_pattern(sample_str, 'state'):
            return 'state'
        elif self._check_location_pattern(sample_str, 'country'):
            return 'country'

        return 'text'
    
    def _check_location_pattern(self, series, location_type):
        """Check if series matches common location patterns"""
        common_cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
                        'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'London', 'Paris', 'Tokyo']
        common_states = ['California', 'Texas', 'Florida', 'New York', 'Pennsylvania', 'Illinois', 
                        'Ohio', 'Georgia', 'North Carolina', 'Michigan']
        common_countries = ['United States', 'USA', 'US', 'Canada', 'United Kingdom', 'UK', 'Germany', 
                           'France', 'Japan', 'China', 'India', 'Brazil', 'Mexico']
        
        sample = series.head(20).str.upper()
        
        if location_type == 'city':
            return any(city.upper() in sample.values for city in common_cities)
        elif location_type == 'state':
            return any(state.upper() in sample.values for state in common_states)
        elif location_type == 'country':
            return any(country.upper() in sample.values for country in common_countries)
        
        return False

    def _looks_like_date(self, sample_str):
        """Check if string values have date-like structure (not just any separator)."""
        date_patterns = re.compile(
            r'^\d{1,4}[\-/\.]\d{1,2}[\-/\.]\d{1,4}$'                        # 2024-01-15, 01/15/2024, 15.01.2024
            r'|^\d{1,4}[\-/\.]\d{1,2}[\-/\.]\d{1,4}\s+\d{1,2}:\d{2}'       # above with time
            r'|^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}$'                      # March 15, 2024
            r'|^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}$'                        # 15 March 2024
            r'|^\d{1,2}[\-/][A-Za-z]{3,9}[\-/]\d{2,4}$'                    # 15-Mar-2024
        )
        matches = sample_str.apply(lambda x: bool(date_patterns.match(str(x).strip())))
        return matches.sum() > len(sample_str) * 0.5

    def anonymize_column(self, series, data_type, df=None):
        """Anonymize a column based on its detected data type"""
        result = series.copy()
        non_null_mask = series.notna()
        
        if data_type == 'datetime':
            result[non_null_mask] = self._anonymize_datetime(series[non_null_mask])
        elif data_type == 'integer':
            result[non_null_mask] = self._anonymize_integer(series[non_null_mask])
        elif data_type == 'float':
            result[non_null_mask] = self._anonymize_float(series[non_null_mask])
        elif data_type == 'email':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.email())
        elif data_type == 'phone':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.phone_number())
        elif data_type == 'name':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.name())
        elif data_type == 'address':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.street_address())
        elif data_type == 'city':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.city())
        elif data_type == 'state':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.state())
        elif data_type == 'state_code':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.state_abbr())
        elif data_type == 'country':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.country())
        elif data_type == 'country_code':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.country_code())
        elif data_type == 'postal_code':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.postcode())
        elif data_type == 'company':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.company())
        elif data_type == 'latitude':
            result[non_null_mask] = self._anonymize_latitude(series[non_null_mask], df)
        elif data_type == 'longitude':
            result[non_null_mask] = self._anonymize_longitude(series[non_null_mask], df)
        elif data_type == 'id':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.uuid4())
        elif data_type == 'boolean':
            # Preserve the same set of values, just shuffle them
            values = series[non_null_mask].values.copy()
            np.random.shuffle(values)
            result[non_null_mask] = values
        elif data_type == 'categorical':
            # Shuffle categorical values to preserve distribution
            values = series[non_null_mask].values.copy()
            np.random.shuffle(values)
            result[non_null_mask] = values
        elif data_type == 'currency':
            result[non_null_mask] = self._anonymize_currency(series[non_null_mask])
        elif data_type == 'url':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.url())
        elif data_type == 'file_path':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.file_path())
        elif data_type == 'ip_address':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.ipv4())
        elif data_type == 'ssn':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.ssn())
        elif data_type == 'credit_card':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.credit_card_number())
        elif data_type == 'username':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.user_name())
        elif data_type == 'job_title':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.job())
        elif data_type == 'license_plate':
            result[non_null_mask] = series[non_null_mask].apply(lambda x: self.fake.license_plate())
        elif data_type == 'date_of_birth':
            result[non_null_mask] = self._anonymize_date_of_birth(series[non_null_mask])
        elif data_type == 'leave_unchanged':
            pass  # return the original data
        else:  # text or unknown
            result[non_null_mask] = self._anonymize_text(series[non_null_mask])
        
        return result
    
    def _anonymize_datetime(self, series):
        """Anonymize datetime values while preserving general range and format"""
        # If already datetime dtype, work directly with timestamps
        if pd.api.types.is_datetime64_any_dtype(series):
            min_date = series.min()
            max_date = series.max()
            date_range = max((max_date - min_date).days, 1)

            result = []
            for _ in series:
                random_days = random.randint(0, date_range)
                new_date = min_date + timedelta(days=random_days)
                result.append(new_date)
            return pd.Series(result, index=series.index)

        # For string dates, detect the format and preserve it
        sample = series.dropna().astype(str).iloc[0]
        detected_format = self._detect_date_format(sample)

        parsed = pd.to_datetime(series)
        min_date = parsed.min()
        max_date = parsed.max()
        date_range = max((max_date - min_date).days, 1)

        result = []
        for _ in series:
            random_days = random.randint(0, date_range)
            new_date = min_date + timedelta(days=random_days)
            if detected_format:
                result.append(new_date.strftime(detected_format))
            else:
                result.append(new_date)
        return pd.Series(result, index=series.index)

    def _detect_date_format(self, sample_str):
        """Detect the date format from a sample string"""
        formats = [
            ('%Y-%m-%d', r'^\d{4}-\d{1,2}-\d{1,2}$'),
            ('%d/%m/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
            ('%m/%d/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
            ('%d-%m-%Y', r'^\d{1,2}-\d{1,2}-\d{4}$'),
            ('%m-%d-%Y', r'^\d{1,2}-\d{1,2}-\d{4}$'),
            ('%Y/%m/%d', r'^\d{4}/\d{1,2}/\d{1,2}$'),
            ('%d.%m.%Y', r'^\d{1,2}\.\d{1,2}\.\d{4}$'),
            ('%B %d, %Y', r'^[A-Z][a-z]+ \d{1,2}, \d{4}$'),
            ('%b %d, %Y', r'^[A-Z][a-z]{2} \d{1,2}, \d{4}$'),
            ('%d %B %Y', r'^\d{1,2} [A-Z][a-z]+ \d{4}$'),
            ('%d %b %Y', r'^\d{1,2} [A-Z][a-z]{2} \d{4}$'),
            ('%Y-%m-%d %H:%M:%S', r'^\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{2}:\d{2}$'),
            ('%m/%d/%Y %H:%M', r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$'),
            ('%d/%m/%Y %H:%M', r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$'),
        ]
        sample_str = str(sample_str).strip()
        for fmt, pattern in formats:
            if re.match(pattern, sample_str):
                try:
                    datetime.strptime(sample_str, fmt)
                    return fmt
                except ValueError:
                    continue
        return None
    
    def _anonymize_integer(self, series):
        """Anonymize integer values while preserving general range"""
        min_val = series.min()
        max_val = series.max()
        
        # Add some variance to the range
        range_size = max_val - min_val
        variance = range_size * 0.1  # 10% variance
        
        result = []
        for _ in series:
            new_val = random.randint(
                int(min_val - variance),
                int(max_val + variance)
            )
            result.append(new_val)
        
        return pd.Series(result, index=series.index)
    
    def _anonymize_float(self, series):
        """Anonymize float values while preserving general range"""
        min_val = series.min()
        max_val = series.max()
        
        # Add some variance to the range
        range_size = max_val - min_val
        variance = range_size * 0.1  # 10% variance
        
        # Determine decimal places
        decimals = series.astype(str).str.extract(r'\.(\d+)')[0].str.len().max()
        if pd.isna(decimals):
            decimals = 2
        
        result = []
        for _ in series:
            new_val = random.uniform(
                min_val - variance,
                max_val + variance
            )
            result.append(round(new_val, int(decimals)))
        
        return pd.Series(result, index=series.index)
    
    def _anonymize_latitude(self, series, df):
        """Anonymize latitude coordinates"""
        result = []
        for idx in series.index:
            # Check if we have a corresponding location in the dataframe
            location_key = self._get_location_key(df, idx) if df is not None else None
            
            if location_key and location_key in self.location_cache:
                lat, lon = self.location_cache[location_key]
                result.append(lat)
            else:
                lat = self.fake.latitude()
                result.append(float(lat))
                
                if location_key:
                    # Store in cache (will be completed when longitude is processed)
                    self.location_cache[location_key] = (float(lat), None)
        
        return pd.Series(result, index=series.index)
    
    def _anonymize_longitude(self, series, df):
        """Anonymize longitude coordinates"""
        result = []
        for idx in series.index:
            # Check if we have a corresponding location in the dataframe
            location_key = self._get_location_key(df, idx) if df is not None else None
            
            if location_key and location_key in self.location_cache and self.location_cache[location_key][1] is not None:
                lat, lon = self.location_cache[location_key]
                result.append(lon)
            else:
                lon = self.fake.longitude()
                result.append(float(lon))
                
                if location_key and location_key in self.location_cache:
                    # Update cache with longitude
                    lat = self.location_cache[location_key][0]
                    self.location_cache[location_key] = (lat, float(lon))
        
        return pd.Series(result, index=series.index)
    
    def _get_location_key(self, df, idx):
        """Generate a key for location caching based on address components"""
        if df is None:
            return None
        
        location_parts = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['city', 'state', 'country', 'address']):
                if idx in df.index:
                    location_parts.append(str(df.loc[idx, col]))
        
        return '|'.join(location_parts) if location_parts else None
    
    def _anonymize_date_of_birth(self, series):
        """Anonymize dates of birth with realistic ages (18-90 years)"""
        today = datetime.today()

        # Detect format if string dates
        is_native_dt = pd.api.types.is_datetime64_any_dtype(series)
        detected_format = None
        if not is_native_dt:
            sample = series.dropna().astype(str).iloc[0]
            detected_format = self._detect_date_format(sample)

        result = []
        for _ in series:
            age_days = random.randint(18 * 365, 90 * 365)
            dob = today - timedelta(days=age_days)
            if detected_format:
                result.append(dob.strftime(detected_format))
            else:
                result.append(pd.Timestamp(dob))
        return pd.Series(result, index=series.index)

    def _anonymize_currency(self, series):
        """Anonymize currency values while preserving format and symbol"""
        result = []
        for val in series:
            s = str(val).strip()
            # Extract the currency symbol and number
            match = re.match(r'^([\$€£¥₹]\s?)([\d,]+\.?\d*)$', s)
            if not match:
                match = re.match(r'^([\d,]+\.?\d*)(\s?[\$€£¥₹])$', s)
                if match:
                    num_str, symbol = match.group(1), match.group(2)
                    suffix = True
                else:
                    result.append(s)
                    continue
            else:
                symbol, num_str = match.group(1), match.group(2)
                suffix = False

            num = float(num_str.replace(',', ''))
            variance = num * 0.3
            new_num = max(0, random.uniform(num - variance, num + variance))

            # Preserve comma formatting and decimal places
            has_decimals = '.' in num_str
            if has_decimals:
                dec_places = len(num_str.split('.')[1])
                formatted = f"{new_num:,.{dec_places}f}"
            else:
                formatted = f"{int(new_num):,}"

            if suffix:
                result.append(f"{formatted}{symbol}")
            else:
                result.append(f"{symbol}{formatted}")

        return pd.Series(result, index=series.index)

    def _anonymize_text(self, series):
        """Anonymize general text data"""
        # For general text, replace with random words of similar length
        result = []
        for text in series:
            if pd.isna(text):
                result.append(text)
            else:
                words = str(text).split()
                new_words = []
                for word in words:
                    # Keep similar word length
                    length = len(word)
                    if length <= 3:
                        new_words.append(self.fake.lexify('?' * length))
                    else:
                        new_words.append(self.fake.word()[:length])
                result.append(' '.join(new_words))
        
        return pd.Series(result, index=series.index)

def main():
    st.title("Data Anonymizer")
    st.markdown("""
    Upload an Excel file to anonymize all data while preserving data types and general patterns.
    The app will intelligently detect data types and apply appropriate anonymization strategies.

    **Ideal for preparing data before sharing with AI tools** — replace sensitive information with
    realistic fake data so you can leverage AI assistance without exposing personal or confidential records.
    """)

    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        try:
            with st.spinner("Reading Excel file..."):
                df = pd.read_excel(uploaded_file)
            st.success(f"Successfully loaded file with {len(df)} rows and {len(df.columns)} columns")

            # Display original data preview
            st.subheader("Original Data Preview (first 5 rows)")
            st.dataframe(df.head())

            # --- Step 1: Analyze ---
            if st.button("Analyze Data Types"):
                anonymizer = DataAnonymizer()
                with st.spinner("Detecting data types..."):
                    detected = {}
                    for col in df.columns:
                        detected[col] = anonymizer.detect_data_type(df[col])
                # Store in session state so it persists across reruns
                st.session_state['detected_types'] = detected
                st.session_state['analysis_done'] = True

            # --- Step 2: Show type overrides and anonymize button ---
            if st.session_state.get('analysis_done'):
                detected = st.session_state['detected_types']

                st.subheader("Review & Adjust Data Types")
                st.caption("Adjust any misdetected types using the dropdowns, then click Anonymize.")

                # Build the override grid
                user_types = {}
                for col in df.columns:
                    c1, c2, c3 = st.columns([2, 2, 3])
                    with c1:
                        st.markdown(f"**{col}**")
                    with c2:
                        default_idx = ALL_DATA_TYPES.index(detected[col]) if detected[col] in ALL_DATA_TYPES else 0
                        user_types[col] = st.selectbox(
                            f"Type for {col}",
                            ALL_DATA_TYPES,
                            index=default_idx,
                            key=f"dtype_{col}",
                            label_visibility="collapsed",
                        )
                    with c3:
                        sample = ', '.join(df[col].dropna().astype(str).head(3).tolist())
                        st.text(sample[:80] + ('...' if len(sample) > 80 else ''))

                # --- Step 3: Anonymize with confirmed types ---
                if st.button("Anonymize Data"):
                    anonymizer = DataAnonymizer()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    anonymized_df = df.copy()
                    total = len(df.columns)
                    for i, col in enumerate(df.columns):
                        status_text.text(f"Anonymizing column: {col}")
                        anonymized_df[col] = anonymizer.anonymize_column(
                            df[col],
                            user_types[col],
                            df,
                        )
                        progress_bar.progress((i + 1) / total)

                    progress_bar.progress(1.0)
                    status_text.text("Anonymization complete!")

                    # Display anonymized data preview
                    st.subheader("Anonymized Data Preview (first 5 rows)")
                    st.dataframe(anonymized_df.head())

                    # Prepare download
                    with st.spinner("Preparing Excel file for download..."):
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            anonymized_df.to_excel(writer, index=False)
                        output.seek(0)

                    st.download_button(
                        label="Download Anonymized Excel File",
                        data=output,
                        file_name=f"anonymized_{uploaded_file.name}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                    # Display statistics
                    st.subheader("Anonymization Statistics")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Total Rows", len(df))
                    with c2:
                        st.metric("Total Columns", len(df.columns))
                    with c3:
                        st.metric("Data Types Used", len(set(user_types.values())))

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    main()