import logging
from datetime import date
from typing import List, Dict, Optional, Any
import pandas as pd

try:
    import gspread
    from gspread.spreadsheet import Spreadsheet
    from gspread.worksheet import Worksheet
except ImportError:
    gspread = None

logger = logging.getLogger(__name__)

class GoogleSheetsClient:
    """Client for pushing portfolio data to Google Sheets."""

    def __init__(self, credentials_path: str = "service_account.json"):
        if not gspread:
            logger.error("gspread library not installed. Install with: pip install gspread")
            self.client = None
            return

        try:
            self.client = gspread.service_account(filename=credentials_path)
            logger.info(f"Authenticated with Google Sheets using {credentials_path}")
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Sheets: {e}")
            self.client = None

    def get_or_create_spreadsheet(self, title: str) -> Optional[Any]:
        """Open existing spreadsheet or create a new one."""
        if not self.client:
            return None
            
        try:
            # Try to open existing
            sheet = self.client.open(title)
            logger.info(f"Opened existing spreadsheet: {title}")
            return sheet
        except gspread.SpreadsheetNotFound:
            try:
                # Create new
                sheet = self.client.create(title)
                logger.info(f"Created new spreadsheet: {title}")
                # Share with the service account's email is automatic, but usually user wants it shared with them.
                # We can't easily know the user's email here without config.
                # For now, we just create it.
                return sheet
            except Exception as e:
                logger.error(f"Failed to create spreadsheet {title}: {e}")
                return None

    def push_portfolio(self, spreadsheet_title: str, tab_name: str, 
                       picks: List[Dict[str, Any]], 
                       headers: List[str] = None) -> bool:
        """
        Push portfolio picks to a specific tab in the spreadsheet.
        
        Args:
            spreadsheet_title: Name of the Google Sheet file (e.g. "BIST Portfolio Tracker")
            tab_name: Name of the worksheet tab (e.g. "Apr 2023 - Alpha")
            picks: List of dicts containing the row data
            headers: Optional list of column headers. If None, derived from keys of first pick.
        """
        if not self.client:
            logger.warning("No active Google Sheets client. Skipping push.")
            return False

        if not picks:
            logger.warning("No picks provided to push.")
            return False

        sheet = self.get_or_create_spreadsheet(spreadsheet_title)
        if not sheet:
            return False

        # Prepare DataFrame for easy formatting
        df = pd.DataFrame(picks)
        
        # If headers not provided, use columns
        if not headers:
            headers = df.columns.tolist()
            
        # Ensure df aligns with headers if possible, or just use what we have
        # Data preparation: convert everything to string or float for sheets
        data_rows = [headers] + df[headers].values.tolist()

        try:
            # Check if tab exists
            try:
                worksheet = sheet.worksheet(tab_name)
                logger.info(f"Found existing tab: {tab_name}")
                worksheet.clear() # Clear existing content
            except gspread.WorksheetNotFound:
                worksheet = sheet.add_worksheet(title=tab_name, rows=100, cols=20)
                logger.info(f"Created new tab: {tab_name}")

            # Update data
            worksheet.update(values=data_rows, range_name="A1")
            
            # Basic Formatting (Bold headers)
            worksheet.format("A1:Z1", {"textFormat": {"bold": True}})
            
            logger.info(f"Successfully pushed {len(picks)} rows to {spreadsheet_title} / {tab_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to push data to sheet: {e}")
            return False
