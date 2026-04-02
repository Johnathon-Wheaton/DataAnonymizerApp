# Data Anonymizer

A Streamlit web application that intelligently anonymizes Excel data while preserving data types and realistic value ranges. Ideal for preparing data before sharing with AI tools — replace sensitive information with realistic fake data so you can leverage AI assistance without exposing personal or confidential records.

## Features

- **Smart Data Type Detection**: Automatically detects 30+ data types including dates, numbers, emails, addresses, coordinates, SSNs, credit cards, and more
- **User-Overridable Types**: Review detected types and adjust them via dropdown menus before anonymizing
- **Intelligent Anonymization**: Preserves data characteristics while replacing with fake but realistic values
- **Range Preservation**: Maintains approximate ranges for numeric, date, and currency data
- **Format Preservation**: Detects and preserves date formats (e.g. `YYYY-MM-DD`, `DD/MM/YYYY`, `March 15, 2024`)
- **Location Coordination**: Coordinates latitude/longitude pairs when anonymizing geographic data
- **Boolean & Categorical Support**: Shuffles low-cardinality columns to preserve value distributions
- **Currency Handling**: Anonymizes monetary values while preserving symbols, comma formatting, and decimal places
- **Leave Unchanged Option**: Skip anonymization on columns that don't contain sensitive data
- **Loading Indicators**: Spinners and progress bars so you always know the app is working

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run anonymizer_app.py
```

## Usage

1. Open the app in your browser (typically http://localhost:8501)
2. Upload an Excel file (.xlsx or .xls)
3. Click **"Analyze Data Types"** to detect column types
4. Review the detected types and adjust any using the dropdown menus
5. Click **"Anonymize Data"** to generate the anonymized dataset
6. Preview the results and download the anonymized Excel file

## Supported Data Types

- **Personal Info**: Names, emails, phone numbers, usernames, SSNs, dates of birth
- **Financial**: Currency values, credit card numbers
- **Geographic**: Addresses, cities, states, countries, postal codes, latitude/longitude
- **Business**: Company names, job titles
- **Numeric**: Integers and floats (maintains approximate range)
- **Dates**: Date/datetime values in any common format (maintains date range)
- **Technical**: URLs, file paths, IP addresses, IDs, license plates
- **Categorical**: Boolean and low-cardinality columns (shuffled to preserve distribution)
- **Text**: General text content
- **Leave Unchanged**: Pass through columns that don't need anonymization

## How It Works

1. **Detect Data Types**: Analyzes column names, data patterns, and value distributions
2. **User Review**: Presents detected types with dropdowns for manual override
3. **Preserve Ranges**: For numeric and date data, generates new values within similar ranges
4. **Coordinate Geography**: Links related geographic fields for consistency
5. **Generate Realistic Data**: Uses the Faker library to create plausible replacement values

## Example

Original data:
```
Name        | Email           | Salary    | City      | Date
John Smith  | john@email.com  | $75,000   | New York  | 2023-01-15
```

Anonymized data:
```
Name        | Email             | Salary    | City      | Date
Jane Doe    | jane@example.org  | $82,340   | Chicago   | 2023-03-22
```
