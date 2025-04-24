# data_loader.py
import streamlit as st
import pandas as pd
import warnings
import gspread
from google.oauth2.service_account import Credentials
import os
import traceback
import re # Import regex for robust key extraction

from config import (
    # Constants for auth fallback and type conversion remain
    LOCAL_KEY_PATH, SCOPES,
    NUMERIC_COLS_CONFIG, DATE_COLS_CONFIG
)

warnings.filterwarnings("ignore")

# Helper function to extract key from URL
def extract_sheet_key(url):
    """Extracts the Google Sheet key from various URL formats."""
    # Regex to find the key between /d/ and /edit or end of string
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if match:
        return match.group(1)
    else:
        # Fallback or error if needed, maybe raise an error
        # For simplicity, returning None here, will be caught later
        return None


@st.cache_data(ttl=18000, show_spinner="Fetching data from Google Sheet...")
def load_data_from_gsheet():
    """Loads data from the specified Google Sheet worksheet using secrets."""

    # --- DIAGNOSTIC PRINT ---
    # print("-" * 20)
    # print("DEBUG: Loaded st.secrets keys:", st.secrets.keys())
    # print("-" * 20)
    # --- END DIAGNOSTIC PRINT ---

    creds = None
    secrets_used = False

    # --- Authentication ---
    # Tries secrets first, then local file.
    try:
        # Check if secrets exist and contain the GCP key
        if hasattr(st, 'secrets') and "gcp_service_account" in st.secrets:
            creds_json_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_json_dict, scopes=SCOPES)
            secrets_used = True
            # st.info("Using GCP credentials from Streamlit secrets.")
    except FileNotFoundError:
        # This means secrets.toml doesn't exist locally, proceed to local file check
        pass # Indentation fixed here (was potentially missing)
    except Exception as e_secrets:
        # Catch other potential errors during secrets processing
        st.warning(f"Error processing Streamlit secrets: {e_secrets}. Trying local key file.")
        pass # Indentation fixed here (was potentially missing)

    # Fallback to local JSON file if secrets weren't successfully used or available
    if not secrets_used:
        if os.path.exists(LOCAL_KEY_PATH):
            try:
                creds = Credentials.from_service_account_file(LOCAL_KEY_PATH, scopes=SCOPES)
                # st.info(f"Using GCP credentials from local file: '{LOCAL_KEY_PATH}'.")
            except Exception as e_local:
                # Handle errors loading from the local file specifically
                st.error(f"Error loading credentials from local file '{LOCAL_KEY_PATH}': {e_local}")
                st.stop() # Stop if local file exists but can't be read
        else:
            # This error occurs if secrets failed AND local file doesn't exist
            st.error(f"Authentication Error: GCP credentials not found in Streamlit Secrets and local key file '{LOCAL_KEY_PATH}' not found.")
            st.info("For deployment, add [gcp_service_account] section to secrets.toml. For local use, ensure service_account.json exists.")
            st.stop() # Stop execution if no credentials found

    # Final check if credentials object was successfully created by either method
    if not creds:
        st.error("Authentication failed. Could not load credentials object.")
        st.stop()

    # --- Authorize and Open Sheet ---
    try:
        client = gspread.authorize(creds)
        sheet_key = None # Initialize sheet_key
        worksheet_name = None # Initialize worksheet_name
        spreadsheet = None # Initialize spreadsheet

        try:
            # --- Read URL and Worksheet Name from Secrets ---
            sheet_url = st.secrets["google_sheet_url"]
            worksheet_name = st.secrets["worksheet_name"] # Read worksheet name

            # --- Extract the Key from the URL ---
            sheet_key = extract_sheet_key(sheet_url)
            if not sheet_key:
                st.error(f"Could not extract Google Sheet key from the URL in secrets: {sheet_url}")
                st.stop()

            # --- Open Sheet using Extracted Key ---
            spreadsheet = client.open_by_key(sheet_key)
            # st.success(f"Successfully opened Google Sheet: '{spreadsheet.title}'")

        except KeyError as e:
            # Handle case where keys are missing in secrets.toml
            st.error(f"Error: '{e.args[0]}' not found in Streamlit Secrets (secrets.toml).")
            st.info("Please ensure google_sheet_url and worksheet_name are defined in your secrets file.")
            st.info(f"Available keys found by Streamlit: {st.secrets.keys()}") # Keep debug info
            st.stop()
        except gspread.exceptions.SpreadsheetNotFound:
            st.error(f"Error: Google Sheet with extracted key '{sheet_key}' not found or not shared.")
            st.info(f"Ensure the URL in secrets is correct and the Sheet is shared with: {creds.service_account_email}")
            st.stop()
        except gspread.exceptions.APIError as api_error:
             st.error(f"Google Sheets API Error while opening spreadsheet: {api_error}")
             st.info("Check API permissions and sharing settings.")
             st.stop()
        except Exception as e_open: # Catch any other unexpected error during opening
             st.error(f"An unexpected error occurred while opening the sheet: {e_open}")
             st.error(traceback.format_exc())
             st.stop()


        # --- Open Worksheet using name from Secrets ---
        # Ensure spreadsheet object was created before proceeding
        if spreadsheet is None:
             st.error("Failed to open spreadsheet object. Cannot proceed.")
             st.stop()

        try:
            if not worksheet_name:
                 st.error("Worksheet name could not be read from secrets.")
                 st.stop()
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            st.error(f"Error: Worksheet '{worksheet_name}' not found in the spreadsheet '{spreadsheet.title}'.")
            st.info("Check the worksheet_name in secrets.toml matches the actual tab name.")
            st.stop()
        except Exception as e_worksheet: # Catch any other error opening worksheet
            st.error(f"An error occurred opening worksheet '{worksheet_name}': {e_worksheet}")
            st.error(traceback.format_exc())
            st.stop()


        # --- Read Data and Convert Types (Moved inside the main try block) ---
        # This ensures df is only processed if sheet/worksheet opening succeeded
        try:
            data = worksheet.get_all_values()
            if not data or len(data) < 2:
                st.warning(f"No data found in worksheet '{worksheet_name}' or only headers present.")
                return pd.DataFrame() # Return empty DataFrame

            headers = data.pop(0)
            df = pd.DataFrame(data, columns=headers) # 'df' is defined here

            # --- Data Type Conversion section (uses NUMERIC/DATE_COLS_CONFIG) ---
            # Now this code only runs if 'df' was successfully created
            numeric_cols = [col for col in NUMERIC_COLS_CONFIG if col in df.columns]
            date_cols = [col for col in DATE_COLS_CONFIG if col in df.columns]

            for col in numeric_cols:
                # Check if column exists before processing (belt-and-suspenders)
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(r'[£,]', '', regex=True).str.strip()
                    df[col] = df[col].replace('', pd.NA)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            for col in date_cols:
                 if col in df.columns:
                    df[col] = df[col].replace('', pd.NaT)
                    df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)

            df = df.replace('', None)
            # st.success("Data loaded and types converted successfully.")
            return df # Return the processed DataFrame

        except Exception as e_read: # Catch errors during reading/processing
            st.error(f"An error occurred reading or processing data from worksheet '{worksheet_name}': {e_read}")
            st.error(traceback.format_exc())
            st.stop()


    # --- Outer Exception Handling ---
    # Catch errors that might occur during client authorization itself
    except gspread.exceptions.APIError as e_api:
        st.error(f"Google Sheets API Error during client authorization or initial access: {e_api}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during Google Sheets access setup: {e}")
        st.error(traceback.format_exc())
        st.stop()

