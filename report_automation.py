
import pandas as pd
import logging
from datetime import datetime, timedelta
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill, Border, Side, Alignment, Font
from openpyxl.workbook import Workbook
from copy import copy
import os
import time
import win32com.client
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='report_automation.log'
)
logger = logging.getLogger('report_automation')

# --- Part 1: Web Automation ---
# Set up Chrome WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
driver.maximize_window()  # Maximize the browser window

DOWNLOADS_FOLDER = "C:/Users/SaiAryanNampally/Downloads"

# Function to get the latest file(s) in the Downloads folder
def get_latest_files(folder, before_files, max_wait=60):
    """Wait for download to complete and return new files, excluding .crdownload files"""
    wait_time = 0
    while wait_time < max_wait:
        time.sleep(2)
        wait_time += 2
        after_files = set(os.listdir(folder))
        new_files = after_files - before_files
        
        # Filter out .crdownload files (incomplete downloads)
        complete_files = [f for f in new_files if not f.endswith('.crdownload')]
        
        if complete_files:
            return [os.path.join(folder, f) for f in complete_files]
    
    # If we've waited too long, return None
    return None

try:
    # Open Ivanti login page
    logger.info("Navigating to Ivanti login page")
    driver.get("https://enwl.saasiteu.com/Account/Login?NoDefaultProvider=true")

    # Enter username and password
    logger.info("Entering username and password")
    username = driver.find_element(By.ID, "UserName")
    username.send_keys("sai.nampally@nxzen.com")
    password = driver.find_element(By.ID, "Password")
    password.send_keys("Summer2024?1")
    login_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]")
    login_button.click()

    # Wait for dashboard and select role
    wait = WebDriverWait(driver, 30)
    logger.info("Waiting for role selection page")
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Service Desk Analyst New')]")))
    role = driver.find_element(By.XPATH, "//*[contains(text(), 'Service Desk Analyst New')]")
    role.click()

    # Click the Submit button if required
    logger.info("Clicking Submit button after role selection")
    submit_button = driver.find_element(By.XPATH, "//*[contains(text(), 'Submit')]")
    submit_button.click()

    # Pause to inspect the page after role selection
    logger.info("Pausing after role selection. Inspect the page now.")
    print("Pausing after role selection. Inspect the page now.")
    wait_short = WebDriverWait(driver, 15)

    # Navigate to Incident and export
    logger.info("Navigating to Incident tab")
    incident = wait_short.until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Incident')]")))
    incident.click()

    # Pause to inspect the page after clicking Incident
    logger.info("Pausing after clicking Incident. Inspect the page now.")
    print("Pausing after clicking Incident. Inspect the page now.")
    time.sleep(8)

    # Check for iframes and switch to the one containing the grid
    logger.info("Checking for iframes on the page")
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    logger.info(f"Found {len(iframes)} iframes on the page")
    team_column = None

    for iframe in iframes:
        try:
            driver.switch_to.frame(iframe)
            logger.info(f"Switched to iframe: {iframe.get_attribute('id') or iframe.get_attribute('name')}")
            team_column = wait_short.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'x-grid3-hd-inner') and contains(., 'Team')]//a[contains(@class, 'x-grid3-hd-btn')]")))
            logger.info("Found Team column header inside iframe")
            break
        except Exception as e:
            logger.warning(f"Could not find Team column in iframe: {str(e)}")
            driver.switch_to.default_content()

    if not team_column:
        try:
            driver.switch_to.frame("ext-gen98")
            logger.info("Switched to iframe with ID 'ext-gen98'")
            team_column = wait_short.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'x-grid3-hd-inner') and contains(., 'Team')]//a[contains(@class, 'x-grid3-hd-btn')]")))
        except Exception as e:
            logger.error(f"Could not find Team column in iframe 'ext-gen98': {str(e)}")
            raise Exception("Could not find Team column in any iframe")

    # Click Team column header to open filter dropdown
    logger.info("Clicking Team column header to open filter dropdown")
    driver.execute_script("arguments[0].scrollIntoView(true);", team_column)
    team_column.click()
    time.sleep(2)

    # Locate and click the Filters dropdown by text
    logger.info("Locating and clicking Filters dropdown by text")
    filters = wait_short.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'x-menu-item x-menu-item-arrow')]/span[text()='Filters']")))
    driver.execute_script("arguments[0].scrollIntoView(true);", filters)
    filters.click()
    time.sleep(2)

    # Locate the row where the first column contains 'SAP - ENZEN'
    row = wait_short.until(EC.presence_of_element_located(
        (By.XPATH, "//td[contains(@class, 'x-grid3-td-0')]//div[text()='SAP - ENZEN']/ancestor::tr")
    ))

    # Click the checkbox in the same row
    checkbox = row.find_element(By.XPATH, ".//td[contains(@class, 'x-grid3-check-col-td')]//div[contains(@class, 'x-grid3-cc-')]")
    checkbox.click()

    time.sleep(5)

    # Export to Excel
    logger.info("Exporting Incident data to Excel")
    export_button = wait_short.until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Export to Excel')]")))
    export_button.click()
    time.sleep(5)

    # Handle the first confirmation popup
    logger.info("Waiting for first confirmation popup")
    yes_button_1 = wait_short.until(EC.element_to_be_clickable((By.XPATH, f"//button[contains(@class, ' x-btn-text') and contains(text(), 'Yes')]")))
    yes_button_1.click()
    logger.info("Clicked Yes on first confirmation popup")
    time.sleep(5)

    # Handle the second confirmation popup
    logger.info("Waiting for second confirmation popup")
    yes_button_2 = wait_short.until(EC.element_to_be_clickable((By.XPATH, f"//button[contains(@class, ' x-btn-text') and contains(text(), 'Yes')]")))
    before_downloads_incident = set(os.listdir(DOWNLOADS_FOLDER))
    yes_button_2.click()
    logger.info("Clicked Yes on second confirmation popup")

    logger.info("Waiting for incident file download to complete...")
    incident_files = get_latest_files(DOWNLOADS_FOLDER, before_downloads_incident, max_wait=60)
    if not incident_files:
        raise Exception("Incident Report Download failed: No new files found!")
    if len(incident_files) != 1:
        raise Exception(f"Incident Report Download failed: Expected 1 file, but found {len(incident_files)} files: {incident_files}")
    incident_file = incident_files[0]
    print(f"Incident Report Downloaded: {incident_file}")

    # Switch back to main content before interacting with Task tab
    driver.switch_to.default_content()
    logger.info("Switched back to main content")

    # Close the Incident tab
    logger.info("Closing the Incident tab")
    close_inc = wait_short.until(EC.element_to_be_clickable((By.XPATH, "//li[contains(@class, 'x-tab-strip') and .//span[contains(text(), 'Incident')]]//a[contains(@class, 'x-tab-strip-close')]")))
    close_inc.click()
    time.sleep(2)

    # Navigate to Task and export
    logger.info("Navigating to Task tab")
    task = wait_short.until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Task')]")))
    task.click()
    logger.info("Waiting for Task page to load completely...")
    time.sleep(45)  # Increased wait time for page to fully load

    # Find the correct iframe dynamically for Task tab
    logger.info("Checking for iframes on Task page")
    max_attempts = 3
    team_column = None
    task_iframe_found = False
    
    for attempt in range(max_attempts):
        logger.info(f"Attempt {attempt + 1} to find Task iframe")
        task_iframes = driver.find_elements(By.TAG_NAME, "iframe")
        logger.info(f"Found {len(task_iframes)} iframes on the Task page")
        
        for i, iframe in enumerate(task_iframes):
            try:
                # Get iframe info before switching
                try:
                    iframe_id = iframe.get_attribute('id') or iframe.get_attribute('name') or f'iframe_{i}'
                except:
                    iframe_id = f'iframe_{i}'
                
                logger.info(f"Attempting to switch to iframe: {iframe_id}")
                driver.switch_to.default_content()
                driver.switch_to.frame(i)  # Use index instead of element reference
                logger.info(f"Successfully switched to iframe {iframe_id}")
                
                # First, try to click SD button if it exists
                try:
                    dropdown_button = driver.find_element(By.XPATH, "//td[contains(@class, 'x-toolbar-cell')]//button[contains(@class, 'x-btn-text') and text()='SD']")
                    dropdown_button.click()
                    logger.info("Clicked on the 'SD' button inside iframe")
                    time.sleep(2)

                    all_option = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                        (By.XPATH, "//div[contains(@class, 'x-saved-search-name') and text()='All']")
                    ))
                    all_option.click()
                    logger.info("Selected 'All' from the dropdown")
                    time.sleep(15)  # Wait for grid to reload
                except Exception as sd_error:
                    logger.info(f"SD button not found or already set in iframe {iframe_id}: {str(sd_error)}")
                
                # Now try to find the Team column
                try:
                    team_column = driver.find_element(By.XPATH, "//div[contains(@class, 'x-grid3-hd-inner') and contains(., 'Team')]//a[contains(@class, 'x-grid3-hd-btn')]")
                    logger.info(f"Found Team column in iframe: {iframe_id}")
                    task_iframe_found = True
                    break
                except Exception as team_error:
                    logger.warning(f"Could not find Team column in iframe {iframe_id}: {str(team_error)}")
                    driver.switch_to.default_content()
                    continue
                    
            except Exception as e:
                logger.warning(f"Error with iframe {i}: {str(e)}")
                try:
                    driver.switch_to.default_content()
                except:
                    pass
                continue
        
        if task_iframe_found:
            break
        
        # If not found, wait and try again
        if attempt < max_attempts - 1:
            logger.info("Team column not found, waiting before retry...")
            driver.switch_to.default_content()
            time.sleep(10)
    
    if not task_iframe_found:
        logger.error("Could not find Task iframe with Team column after all attempts")
        raise Exception("Could not find Team column in any iframe for Task tab")

    time.sleep(10)

    # Click Team column header to open filter dropdown
    logger.info("Clicking Team column header to open filter dropdown")
    driver.execute_script("arguments[0].scrollIntoView(true);", team_column)
    team_column.click()
    time.sleep(15)

    # Locate and click the Filters dropdown by text
    logger.info("Locating and clicking Filters dropdown by text")
    filters = wait_short.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'x-menu-item x-menu-item-arrow')]/span[text()='Filters']")))
    driver.execute_script("arguments[0].scrollIntoView(true);", filters)
    filters.click()
    time.sleep(2)

    # Locate the row where the first column contains 'SAP - ENZEN'
    row = wait_short.until(EC.presence_of_element_located(
        (By.XPATH, "//td[contains(@class, 'x-grid3-td-0')]//div[text()='SAP - ENZEN']/ancestor::tr")
    ))

    # Click the checkbox in the same row
    checkbox = row.find_element(By.XPATH, ".//td[contains(@class, 'x-grid3-check-col-td')]//div[contains(@class, 'x-grid3-cc-')]")
    checkbox.click()

    time.sleep(5)

    # Export to Excel
    logger.info("Exporting Task data to Excel")
    export_button = wait_short.until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Export Tasks to Excel')]")))
    export_button.click()
    time.sleep(5)

    # Handle the first confirmation popup
    logger.info("Waiting for first confirmation popup")
    yes_button_1 = wait_short.until(EC.element_to_be_clickable((By.XPATH, f"//button[contains(@class, ' x-btn-text') and contains(text(), 'Yes')]")))
    yes_button_1.click()
    logger.info("Clicked Yes on first confirmation popup")
    time.sleep(5)

    # Handle the second confirmation popup
    logger.info("Waiting for second confirmation popup")
    yes_button_2 = wait_short.until(EC.element_to_be_clickable((By.XPATH, f"//button[contains(@class, ' x-btn-text') and contains(text(), 'Yes')]")))
    before_downloads_tasks = set(os.listdir(DOWNLOADS_FOLDER))
    yes_button_2.click()
    logger.info("Clicked Yes on second confirmation popup")

    logger.info("Waiting for tasks file download to complete...")
    tasks_files = get_latest_files(DOWNLOADS_FOLDER, before_downloads_tasks, max_wait=60)
    if not tasks_files:
        raise Exception("Tasks Report Download failed: No new files found!")
    if len(tasks_files) != 1:
        raise Exception(f"Tasks Report Download failed: Expected 1 file, but found {len(tasks_files)} files: {tasks_files}")
    tasks_file = tasks_files[0]
    print(f"Tasks Report Downloaded: {tasks_file}")

    # --- Part 2: Excel Processing ---
    try:
        # Process Incident file
        logger.info("Processing Incident file")
        df_incident = pd.read_excel(incident_file)
        logger.info(f"Incident file loaded successfully. Columns: {df_incident.columns.tolist()}")

        # Explicitly define column names
        incident_column_mapping = {
            "created on": "Created On",
            "closed on": "Resolved On",
            "incident": "Incident",
            "owner": "Owner",
            "status": "Status",
            "service": "Service"
        }

        # Verify and map columns
        inc_created_on_col = incident_column_mapping["created on"]
        inc_closed_on_col = incident_column_mapping["closed on"]
        inc_incident_col = incident_column_mapping["incident"]
        inc_owner_col = incident_column_mapping["owner"]
        inc_status_col = incident_column_mapping["status"]
        inc_service_col = incident_column_mapping["service"]

        # Check if columns exist
        missing_inc_cols = [v for k, v in incident_column_mapping.items() if v not in df_incident.columns]
        if missing_inc_cols:
            logger.warning(f"Warning: Columns {missing_inc_cols} not found in incident file! Check mapping.")

        # Add Formatted Date columns
        if inc_created_on_col in df_incident.columns:
            df_incident["Created_Formatted"] = df_incident[inc_created_on_col].apply(
                lambda x: pd.to_datetime(x).strftime("%d-%m-%Y") if pd.notna(x) else ""
            )
        else:
            df_incident["Created_Formatted"] = ""

        if inc_closed_on_col in df_incident.columns:
            df_incident["Closed_Formatted"] = df_incident[inc_closed_on_col].apply(
                lambda x: pd.to_datetime(x).strftime("%d-%m-%Y") if pd.notna(x) else ""
            )
        else:
            df_incident["Closed_Formatted"] = ""

        # Incident Pivot Tables
        yesterday = "19-01-2026"
        incident_pivot_statuses = ["Awaiting 3rd Party Feedback", "Awaiting Customer Feedback", "Work in Progress", "Resolved"]

        # Service Pivot for SAP
        service_pivot_inc = pd.pivot_table(
            df_incident[df_incident[inc_service_col] == "SAP"],
            index=inc_owner_col,
            columns=inc_status_col,
            values=inc_incident_col,
            aggfunc="count",
            fill_value=0
        ).reindex(columns=["Awaiting 3rd Party Feedback", "Awaiting Customer Feedback", "Work in Progress"], fill_value=0)
        service_pivot_inc['Grand Total'] = service_pivot_inc.sum(axis=1)
        service_pivot_inc = service_pivot_inc[service_pivot_inc['Grand Total'] > 0].loc[:, (service_pivot_inc != 0).any()]

        # Created Pivot
        created_filtered_df = df_incident[df_incident["Created_Formatted"] == yesterday]
        created_pivot_inc = pd.pivot_table(
            data=created_filtered_df,
            columns=inc_status_col,
            values=inc_incident_col,
            aggfunc="count",
            fill_value=0
        ).reindex(columns=incident_pivot_statuses, fill_value=0)
        created_pivot_inc['Grand Total'] = created_pivot_inc.sum(axis=1)
        created_pivot_inc = created_pivot_inc[created_pivot_inc['Grand Total'] > 0].loc[:, (created_pivot_inc != 0).any()]

        # Closed Pivot
        closed_pivot_inc = pd.pivot_table(
            df_incident[df_incident["Closed_Formatted"] == yesterday],
            # index=inc_owner_col,
            columns=inc_status_col,
            values=inc_incident_col,
            aggfunc="count",
            fill_value=0
        ).reindex(columns=incident_pivot_statuses, fill_value=0)
        closed_pivot_inc['Grand Total'] = closed_pivot_inc.sum(axis=1)
        closed_pivot_inc = closed_pivot_inc[closed_pivot_inc['Grand Total'] > 0].loc[:, (closed_pivot_inc != 0).any()]

        # --- Process Task file ---
        logger.info("Processing Task file")
        df_task = pd.read_excel(tasks_file)
        logger.info(f"Task file loaded successfully. Columns: {df_task.columns.tolist()}")

        # Explicitly define task column names
        task_column_mapping = {
            "created on": "Created On",
            "completed on": "Completed On",
            "task id#": "Task ID#",
            "owner": "Owner",
            "status": "Status",
            "service": "Service"
        }

        # Verify and map task columns
        task_created_on_col = task_column_mapping["created on"]
        task_completed_on_col = task_column_mapping["completed on"]
        task_id_col = task_column_mapping["task id#"]
        task_owner_col = task_column_mapping["owner"]
        task_status_col = task_column_mapping["status"]
        task_service_col = task_column_mapping["service"]

        # Check if columns exist
        missing_task_cols = [v for k, v in task_column_mapping.items() if v not in df_task.columns]
        if missing_task_cols:
            logger.warning(f"Warning: Columns {missing_task_cols} not found in task file! Check mapping.")

        # Add Formatted Date columns
        if task_created_on_col in df_task.columns:
            df_task["Created_Formatted"] = df_task[task_created_on_col].apply(
                lambda x: pd.to_datetime(x).strftime("%d-%m-%Y") if pd.notna(x) else ""
            )
        else:
            df_task["Created_Formatted"] = ""

        if task_completed_on_col in df_task.columns:
            df_task["Closed_Formatted"] = df_task[task_completed_on_col].apply(
                lambda x: pd.to_datetime(x).strftime("%d-%m-%Y") if pd.notna(x) else ""
            )
        else:
            df_task["Closed_Formatted"] = ""

        # Task Pivot Tables
        task_pivot_statuses = ["Accepted", "Logged", "Waiting - 3rd Party", "Waiting - Customer", "", "Completed"]
        desired_services = ["Business - Non SLA", "SAP", "SAP - Infrastructure", "SAP - Vendor - Non SLA", "Service Desk"]

        # Task Service Pivot
        service_pivot_task = pd.pivot_table(
            df_task[df_task[task_service_col].isin(desired_services)],
            index=task_owner_col,
            columns=task_status_col,
            values=task_id_col,
            aggfunc="count",
            fill_value=0
        ).reindex(columns=["Accepted", "Logged", "Waiting - 3rd Party", "Waiting - Customer"], fill_value=0)
        service_pivot_task['Grand Total'] = service_pivot_task.sum(axis=1)
        service_pivot_task = service_pivot_task[service_pivot_task['Grand Total'] > 0].loc[:, (service_pivot_task != 0).any()]

        # Task Created Pivot
        created_pivot_task = pd.pivot_table(
            df_task[df_task["Created_Formatted"] == yesterday],
            # index=task_owner_col,
            columns=task_status_col,
            values=task_id_col,
            aggfunc="count",
            fill_value=0
        ).reindex(columns=task_pivot_statuses, fill_value=0)
        created_pivot_task['Grand Total'] = created_pivot_task.sum(axis=1)
        created_pivot_task = created_pivot_task[created_pivot_task['Grand Total'] > 0].loc[:, (created_pivot_task != 0).any()]

        # Task Closed Pivot
        closed_pivot_task = pd.pivot_table(
            df_task[df_task["Closed_Formatted"] == yesterday],
            # index=task_owner_col,
            columns=task_status_col,
            values=task_id_col,
            aggfunc="count",
            fill_value=0
        ).reindex(columns=task_pivot_statuses, fill_value=0)
        closed_pivot_task['Grand Total'] = closed_pivot_task.sum(axis=1)
        closed_pivot_task = closed_pivot_task[closed_pivot_task['Grand Total'] > 0].loc[:, (closed_pivot_task != 0).any()]

    except FileNotFoundError as e:
        logger.error(f"Error loading file: {e}. Please check file paths.")
        exit()
    except KeyError as e:
        logger.error(f"Error: Column '{e}' not found. Check column mappings and Excel file headers.")
        exit()
    except Exception as e:
        logger.error(f"An unexpected error occurred during data processing: {e}")
        exit()

    # --- Formatting Function ---
    def format_pivot_table(worksheet, start_row, pivot_table, title, count_label, id_col_name):
        try:
            header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            thin_border_side = Side(style='thin', color="000000")
            thin_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
            thick_border_side = Side(style='thick', color="000000")
            center_alignment = Alignment(horizontal='center', vertical='center')
            left_alignment = Alignment(horizontal='left', vertical='center')
            bold_font = Font(bold=True)

            if pivot_table.empty:
                logger.warning(f"Pivot table '{title}' is empty. Skipping formatting.")
                cell = worksheet.cell(row=start_row, column=1, value=f"{title} - No data for {yesterday}")
                cell.font = Font(italic=True)
                return 1

            n_rows_data = len(pivot_table)
            if 'Grand Total' in pivot_table.columns:
                cols = [col for col in pivot_table.columns if col != 'Grand Total'] + ['Grand Total']
            else:
                cols = pivot_table.columns.tolist()
            n_cols_data = len(cols)

            title_row_idx = start_row
            title_date_str = f"{title} - {datetime.strptime(yesterday, '%d-%m-%Y').strftime('%d-%b-%Y')}"
            cell_title = worksheet.cell(row=title_row_idx, column=1, value=title_date_str)
            worksheet.merge_cells(start_row=title_row_idx, start_column=1, end_row=title_row_idx, end_column=n_cols_data + 1)
            cell_title.alignment = left_alignment

            rows_used = 1

            if title in ["Created", "Closed"]:
                col_sums = pivot_table[cols].sum()

                header_row_idx = start_row + rows_used
                headers = [id_col_name] + cols
                for c_idx, header_text in enumerate(headers):
                    col_num = c_idx + 1
                    cell = worksheet.cell(row=header_row_idx, column=col_num, value=header_text)
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = center_alignment
                    cell.border = thin_border

                rows_used += 1

                total_data_row_idx = start_row + rows_used
                cell_gt_label = worksheet.cell(row=total_data_row_idx, column=1, value="Grand Total")
                cell_gt_label.border = thin_border
                cell_gt_label.font = bold_font
                cell_gt_label.alignment = left_alignment

                for c_idx, col_name in enumerate(cols):
                    col_num = c_idx + 2
                    cell_val = worksheet.cell(row=total_data_row_idx, column=col_num, value=col_sums[col_name])
                    cell_val.border = thin_border
                    cell_val.alignment = center_alignment
                    cell_val.font = bold_font
                    cell_val.number_format = '0'

                rows_used += 1
                data_end_row = total_data_row_idx

            else:
                header_row1_idx = start_row + rows_used
                cell1_1 = worksheet.cell(row=header_row1_idx, column=1, value=count_label)
                cell1_1.fill = header_fill
                cell1_1.font = header_font
                cell1_1.alignment = center_alignment
                cell1_1.border = thin_border

                if n_cols_data > 0:
                    cell1_2 = worksheet.cell(row=header_row1_idx, column=2, value="Status")
                    worksheet.merge_cells(start_row=header_row1_idx, start_column=2, end_row=header_row1_idx, end_column=n_cols_data + 1)
                    cell1_2.fill = header_fill
                    cell1_2.font = header_font
                    cell1_2.alignment = center_alignment
                    for c_idx_merge in range(2, n_cols_data + 2):
                        mc = worksheet.cell(row=header_row1_idx, column=c_idx_merge)
                        mc.border = thin_border

                rows_used += 1

                header_row2_idx = start_row + rows_used
                cell2_1 = worksheet.cell(row=header_row2_idx, column=1, value=id_col_name)
                cell2_1.fill = header_fill
                cell2_1.font = header_font
                cell2_1.alignment = center_alignment
                cell2_1.border = thin_border

                for c_idx, col_name in enumerate(cols):
                    col_num = c_idx + 2
                    cell = worksheet.cell(row=header_row2_idx, column=col_num, value=col_name)
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = center_alignment
                    cell.border = thin_border
                rows_used += 1

                data_start_row_idx = start_row + rows_used
                for r_idx, (index_val, row_data) in enumerate(pivot_table.iterrows()):
                    current_row_idx = data_start_row_idx + r_idx
                    cell_index = worksheet.cell(row=current_row_idx, column=1, value=index_val)
                    cell_index.alignment = left_alignment
                    cell_index.border = thin_border

                    for c_idx, col_name in enumerate(cols):
                        col_num = c_idx + 2
                        cell_data = worksheet.cell(row=current_row_idx, column=col_num, value=row_data[col_name])
                        cell_data.alignment = center_alignment
                        cell_data.border = thin_border
                        cell_data.number_format = '0'

                rows_used += n_rows_data

                total_row_idx = start_row + rows_used
                col_sums = pivot_table[cols].sum()
                cell_gt_label = worksheet.cell(row=total_row_idx, column=1, value="Grand Total")
                cell_gt_label.border = thin_border
                cell_gt_label.font = bold_font
                cell_gt_label.alignment = left_alignment

                for c_idx, col_name in enumerate(cols):
                    col_num = c_idx + 2
                    cell_val = worksheet.cell(row=total_row_idx, column=col_num, value=col_sums[col_name])
                    cell_val.border = thin_border
                    cell_val.alignment = center_alignment
                    cell_val.font = bold_font
                    cell_val.number_format = '0'

                rows_used += 1
                data_end_row = total_row_idx

            min_row_border = start_row + 1
            max_row_border = data_end_row
            min_col_border = 1
            max_col_border = n_cols_data + 1

            for row in worksheet.iter_rows(min_row=min_row_border, max_row=max_row_border, min_col=min_col_border, max_col=max_col_border):
                for cell in row:
                    existing_border = copy(cell.border)
                    if cell.row == min_row_border:
                        existing_border.top = thick_border_side
                    if cell.row == max_row_border:
                        existing_border.bottom = thick_border_side
                    if cell.column == min_col_border:
                        existing_border.left = thick_border_side
                    if cell.column == max_col_border:
                        existing_border.right = thick_border_side
                    cell.border = existing_border

            worksheet.column_dimensions[get_column_letter(1)].width = 25
            for c_idx_width in range(n_cols_data):
                col_num = c_idx_width + 2
                worksheet.column_dimensions[get_column_letter(col_num)].width = 20

            return rows_used

        except Exception as e:
            logger.error(f"Error formatting table '{title}': {e}", exc_info=True)
            return 0

    # --- Send Email ---
    def send_outlook_email():
        try:
            outlook = win32com.client.Dispatch("outlook.application")
            mail = outlook.CreateItem(0)  # 0: olMailItem

            # Email recipients
            to_group = "sai.nampally@nxzen.com"  # Replace with your distribution list
            cc_emails = "mahendra.kondapalliteja@nxzen.com;chakrala.agnivesh@nxzen.com"  # Replace with CC email addresses

            # Email details
            mail.To = to_group
            mail.CC = cc_emails
            mail.Subject = f"ENWL Incident and Task Status as on - {datetime.now().strftime('%d-%b-%Y')}"

            # Attach the Excel file
            output_file = f"C:/Users/SaiAryanNampally/OneDrive - Enzen Global/Desktop/Automation/Task_and_Incident_As_On_{datetime.now().strftime('%d_%b_%Y')}.xlsx"
            mail.Attachments.Add(output_file)
            logger.info(f"Attached Excel file: {output_file}")

            # Predefined body
            body = """
            Hi All,

            Please find the attached Incident and Task status report as of {}.

            """.format(datetime.now().strftime('%d-%b-%Y'))

            # Convert pivot tables to HTML and append to body
            pivot_tables = [
                ("Incident - Service", service_pivot_inc),
                ("Incident - Created", created_pivot_inc),
                ("Incident - Closed", closed_pivot_inc),
                ("Task - Service", service_pivot_task),
                ("Task - Created", created_pivot_task),
                ("Task - Closed", closed_pivot_task)
            ]

            for title, pivot in pivot_tables:
                body += f"\n<h3>{title}</h3>\n"
                body += pivot.to_html(classes="pivot-table", index=True).replace('<th>', '<th style="background-color:#4F81BD;color:white;">').replace('<td>', '<td style="border:1px solid black;">')
                body += "<br><br>"

            body += """
            Regards,
            Sai Aryan Nampally
            """

            mail.HTMLBody = body

            # Send the email
            mail.Send()
            logger.info("Email sent successfully")
            print("Email sent successfully")

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            print(f"Error sending email: {e}")

    # --- Save to Output File with Formatting ---
    output_file = f"C:/Users/SaiAryanNampally/OneDrive - Enzen Global/Desktop/Automation/Task_and_Incident_As_On_{datetime.now().strftime('%d_%b_%Y')}.xlsx"
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            workbook = writer.book

            if "Incident" not in workbook.sheetnames:
                inc_sheet = workbook.create_sheet("Incident")
            else:
                inc_sheet = workbook["Incident"]

            current_row = 1
            rows_written = format_pivot_table(inc_sheet, current_row, service_pivot_inc, "Service", "Count of Incident", "Owner")
            current_row += rows_written + 2
            rows_written = format_pivot_table(inc_sheet, current_row, created_pivot_inc, "Created", "Count of Incident", "Incident")
            current_row += rows_written + 2
            rows_written = format_pivot_table(inc_sheet, current_row, closed_pivot_inc, "Closed", "Count of Incident", "Incident")

            if "Task" not in workbook.sheetnames:
                task_sheet = workbook.create_sheet("Task")
            else:
                task_sheet = workbook["Task"]

            current_row = 1
            rows_written = format_pivot_table(task_sheet, current_row, service_pivot_task, "Service", "Count of Task ID#", "Owner")
            current_row += rows_written + 2
            rows_written = format_pivot_table(task_sheet, current_row, created_pivot_task, "Created", "Count of Task ID#", "Task ID#")
            current_row += rows_written + 2
            rows_written = format_pivot_table(task_sheet, current_row, closed_pivot_task, "Closed", "Count of Task ID#", "Task ID#")

            if "Sheet1" in workbook.sheetnames and not workbook["Sheet1"]._cells:
                del workbook["Sheet1"]
                logger.info("Removed default empty 'Sheet1'.")

        logger.info(f"Report saved successfully as {output_file}")
        print(f"Report saved successfully as {output_file}")

    except Exception as e:
        logger.error(f"An error occurred during Excel writing: {e}", exc_info=True)
        print(f"An error occurred during Excel writing: {e}")
        raise

    # Call email function after the file is saved
    send_outlook_email()

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    print(f"An error occurred: {str(e)}")
    raise

finally:
    # Close browser
    logger.info("Closing browser")
    driver.quit()

    # Delete downloaded incident and task files
    try:
        if 'incident_file' in locals() and os.path.exists(incident_file):
            os.remove(incident_file)
            logger.info(f"Deleted incident file: {incident_file}")
        else:
            logger.warning(f"Incident file not found or already deleted: {incident_file}")

        if 'tasks_file' in locals() and os.path.exists(tasks_file):
            os.remove(tasks_file)
            logger.info(f"Deleted task file: {tasks_file}")
        else:
            logger.warning(f"Task file not found or already deleted: {tasks_file}")
    except Exception as e:
        logger.error(f"Error deleting files: {e}")
        print(f"Error deleting files: {e}")