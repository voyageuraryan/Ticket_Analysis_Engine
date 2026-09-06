"""
Testing Data Extraction from Ivanti
Extracts resolved tickets for model testing and validation.

This script:
1. Logs into Ivanti
2. Navigates to Incident tab
3. Filters by Team (SAP - ENZEN), Status (Resolved), and Resolved Date (Today)
4. Extracts ticket details (Summary, Description, Modified By, etc.)
5. Saves to Excel for classification testing
"""

import pandas as pd
import logging
from datetime import datetime
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='testing_data_extraction.log'
)
logger = logging.getLogger('testing_data_extraction')

# Console handler for real-time feedback
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# Set up Chrome WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
driver.maximize_window()

# Output directory
OUTPUT_DIR = "data/testing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    # ==================== STEP 1: LOGIN ====================
    logger.info("=" * 80)
    logger.info("STEP 1: Logging into Ivanti")
    logger.info("=" * 80)
    
    driver.get("https://enwl.saasiteu.com/Account/Login?NoDefaultProvider=true")
    
    # Enter credentials
    logger.info("Entering username and password")
    username = driver.find_element(By.ID, "UserName")
    username.send_keys("sai.nampally@nxzen.com")
    password = driver.find_element(By.ID, "Password")
    password.send_keys("Summer2024?1")
    login_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]")
    login_button.click()
    
    # Wait for role selection
    wait = WebDriverWait(driver, 30)
    logger.info("Waiting for role selection page")
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Service Desk Analyst New')]")))
    role = driver.find_element(By.XPATH, "//*[contains(text(), 'Service Desk Analyst New')]")
    role.click()
    
    # Submit role
    logger.info("Submitting role selection")
    submit_button = driver.find_element(By.XPATH, "//*[contains(text(), 'Submit')]")
    submit_button.click()
    
    logger.info("✅ Login successful")
    time.sleep(5)
    
    # ==================== STEP 2: NAVIGATE TO INCIDENT ====================
    logger.info("=" * 80)
    logger.info("STEP 2: Navigating to Incident tab")
    logger.info("=" * 80)
    
    wait_short = WebDriverWait(driver, 15)
    incident = wait_short.until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Incident')]")))
    incident.click()
    logger.info("✅ Incident tab opened")
    time.sleep(8)
    
    # ==================== STEP 3: SWITCH TO IFRAME ====================
    logger.info("=" * 80)
    logger.info("STEP 3: Finding and switching to correct iframe")
    logger.info("=" * 80)
    
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    logger.info(f"Found {len(iframes)} iframes on the page")
    
    team_column = None
    for idx, iframe in enumerate(iframes):
        try:
            driver.switch_to.frame(iframe)
            iframe_id = iframe.get_attribute('id') or iframe.get_attribute('name') or f'iframe_{idx}'
            logger.info(f"Trying iframe: {iframe_id}")
            
            team_column = wait_short.until(EC.presence_of_element_located(
                (By.XPATH, "//div[contains(@class, 'x-grid3-hd-inner') and contains(., 'Team')]//a[contains(@class, 'x-grid3-hd-btn')]")
            ))
            logger.info(f"✅ Found Team column in iframe: {iframe_id}")
            break
        except Exception as e:
            logger.debug(f"Team column not in this iframe, trying next...")
            driver.switch_to.default_content()
    
    if not team_column:
        # Try known iframe ID as fallback
        try:
            driver.switch_to.frame("ext-gen98")
            logger.info("Switched to iframe with ID 'ext-gen98'")
            team_column = wait_short.until(EC.presence_of_element_located(
                (By.XPATH, "//div[contains(@class, 'x-grid3-hd-inner') and contains(., 'Team')]//a[contains(@class, 'x-grid3-hd-btn')]")
            ))
            logger.info("✅ Found Team column in ext-gen98 iframe")
        except Exception as e:
            logger.error(f"❌ Could not find Team column in any iframe: {str(e)}")
            raise Exception("Could not find Team column in any iframe")
    
    # ==================== STEP 4: APPLY TEAM FILTER ====================
    logger.info("=" * 80)
    logger.info("STEP 4: Applying Team filter (SAP - ENZEN)")
    logger.info("=" * 80)
    
    driver.execute_script("arguments[0].scrollIntoView(true);", team_column)
    team_column.click()
    time.sleep(2)
    
    # Click Filters dropdown
    logger.info("Opening Filters dropdown")
    filters = wait_short.until(EC.element_to_be_clickable(
        (By.XPATH, "//a[contains(@class, 'x-menu-item x-menu-item-arrow')]/span[text()='Filters']")
    ))
    driver.execute_script("arguments[0].scrollIntoView(true);", filters)
    filters.click()
    time.sleep(2)
    
    # Select SAP - ENZEN
    logger.info("Selecting 'SAP - ENZEN' team")
    row = wait_short.until(EC.presence_of_element_located(
        (By.XPATH, "//td[contains(@class, 'x-grid3-td-0')]//div[text()='SAP - ENZEN']/ancestor::tr")
    ))
    checkbox = row.find_element(By.XPATH, ".//td[contains(@class, 'x-grid3-check-col-td')]//div[contains(@class, 'x-grid3-cc-')]")
    checkbox.click()
    logger.info("✅ Team filter applied: SAP - ENZEN")
    time.sleep(5)
    
    # ==================== STEP 5: APPLY STATUS FILTER ====================
    logger.info("=" * 80)
    logger.info("STEP 5: Applying Status filter (Resolved)")
    logger.info("=" * 80)
    
    # Find Status column header
    logger.info("Locating Status column header")
    status_column = wait_short.until(EC.presence_of_element_located(
        (By.XPATH, "//div[contains(@class, 'x-grid3-hd-inner') and contains(., 'Status')]//a[contains(@class, 'x-grid3-hd-btn')]")
    ))
    
    # Click Status column header
    logger.info("Clicking Status column header")
    driver.execute_script("arguments[0].scrollIntoView(true);", status_column)
    status_column.click()
    time.sleep(2)
    
    # Click Filters dropdown
    logger.info("Opening Filters dropdown for Status")
    filters_status = wait_short.until(EC.element_to_be_clickable(
        (By.XPATH, "//a[contains(@class, 'x-menu-item x-menu-item-arrow')]/span[text()='Filters']")
    ))
    driver.execute_script("arguments[0].scrollIntoView(true);", filters_status)
    filters_status.click()
    time.sleep(2)
    
    # Select Resolved
    logger.info("Selecting 'Resolved' status")
    resolved_row = wait_short.until(EC.presence_of_element_located(
        (By.XPATH, "//td[contains(@class, 'x-grid3-td-0')]//div[text()='Resolved']/ancestor::tr")
    ))
    resolved_checkbox = resolved_row.find_element(
        By.XPATH, ".//td[contains(@class, 'x-grid3-check-col-td')]//div[contains(@class, 'x-grid3-cc-')]"
    )
    resolved_checkbox.click()
    logger.info("✅ Status filter applied: Resolved")
    time.sleep(5)
    
    # ==================== STEP 6: APPLY RESOLVED DATE FILTER ====================
    logger.info("=" * 80)
    logger.info("STEP 6: Applying Resolved Date filter (Today)")
    logger.info("=" * 80)
    
    # Find Resolved Date column header
    logger.info("Locating Resolved Date column header")
    
    # Try multiple possible column names
    resolved_date_column = None
    possible_names = ["Resolved Date", "Closed Date", "Resolved On", "Closed On"]
    
    for col_name in possible_names:
        try:
            resolved_date_column = driver.find_element(
                By.XPATH, f"//div[contains(@class, 'x-grid3-hd-inner') and contains(., '{col_name}')]//a[contains(@class, 'x-grid3-hd-btn')]"
            )
            logger.info(f"✅ Found column: {col_name}")
            break
        except:
            logger.debug(f"Column '{col_name}' not found, trying next...")
            continue
    
    if not resolved_date_column:
        logger.warning("⚠️ Could not find Resolved Date column - skipping date filter")
        logger.warning("⚠️ Will extract all resolved tickets (not just today)")
    else:
        # Click Resolved Date column header
        logger.info("Clicking Resolved Date column header")
        driver.execute_script("arguments[0].scrollIntoView(true);", resolved_date_column)
        resolved_date_column.click()
        time.sleep(2)
        
        # Click Filters dropdown
        logger.info("Opening Filters dropdown for Resolved Date")
        filters_date = wait_short.until(EC.element_to_be_clickable(
            (By.XPATH, "//a[contains(@class, 'x-menu-item x-menu-item-arrow')]/span[text()='Filters']")
        ))
        driver.execute_script("arguments[0].scrollIntoView(true);", filters_date)
        filters_date.click()
        time.sleep(3)
        
        # Try to enter today's date in the date filter UI
        try:
            today_date = datetime.now().strftime("%m/%d/%Y")  # Format: MM/DD/YYYY
            logger.info(f"Attempting to set date filter to: {today_date}")
            
            # Find date input fields using the correct class selector
            # These are grid filter inputs, not regular form inputs
            date_inputs = driver.find_elements(
                By.XPATH, "//input[@type='text' and contains(@class, 'x-grid-filter-string') and @size='10']"
            )
            
            logger.info(f"Found {len(date_inputs)} date input fields")
            
            if len(date_inputs) >= 1:
                # We only need to fill the FIRST field (equals)
                # Don't touch the second field at all - it clears the first one
                
                logger.info("Filling ONLY the first date field (equals)")
                first_date_input = date_inputs[0]
                
                # Use JavaScript to set the value
                driver.execute_script(f"arguments[0].value = '{today_date}';", first_date_input)
                logger.info(f"Set first field to: {today_date}")
                time.sleep(0.5)
                
                # Trigger the filter by clicking on the field and pressing Enter
                # This will apply the filter without touching the second field
                first_date_input.click()
                time.sleep(0.3)
                first_date_input.send_keys(Keys.RETURN)
                time.sleep(2)
                
                logger.info(f"✅ Resolved Date filter applied: {today_date} (equals)")
                time.sleep(3)
                
            else:
                raise Exception(f"Expected at least 1 date input field, found {len(date_inputs)}")
            
        except Exception as date_error:
            logger.warning(f"⚠️ Could not apply date filter: {str(date_error)}")
            logger.warning("⚠️ Continuing without date filter - will extract all resolved tickets")
            # Try to close any open menus by pressing Escape
            try:
                from selenium.webdriver.common.keys import Keys
                driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
                time.sleep(1)
            except:
                pass
    
    # ==================== STEP 7: COUNT TICKETS ====================
    logger.info("=" * 80)
    logger.info("STEP 7: Counting filtered tickets")
    logger.info("=" * 80)
    
    # Wait longer for grid to refresh after filters
    logger.info("Waiting for grid to refresh after applying filters...")
    time.sleep(5)
    
    # Try to get the ticket count from the UI element
    ui_count = None
    try:
        # Try multiple XPath strategies to find the count element
        count_xpaths = [
            "//div[contains(@class, 'xtb-text') and contains(@class, 'x-total-records')]",
            "//div[contains(text(), 'search records')]",
            "//div[@id='ext-comp-3455']",
            "//div[contains(@class, 'x-toolbar-cell')]//div[contains(text(), 'records')]",
        ]
        
        for xpath in count_xpaths:
            try:
                count_element = driver.find_element(By.XPATH, xpath)
                count_text = count_element.text  # e.g., "(1 search records)"
                logger.info(f"📊 Ticket count from UI: {count_text}")
                # Extract number from text like "(1 search records)"
                import re
                match = re.search(r'\((\d+)\s+search', count_text)
                if match:
                    ui_count = int(match.group(1))
                    logger.info(f"📊 Parsed UI count: {ui_count}")
                break
            except:
                continue
    except Exception as e:
        logger.warning(f"⚠️ Could not read ticket count from UI element: {str(e)}")
    
    # Get all ticket rows (visible and non-empty)
    all_rows = driver.find_elements(
        By.XPATH, "//div[contains(@class, 'x-grid3-body')]//div[contains(@class, 'x-grid3-row')]"
    )
    
    # Filter to only visible rows with content
    ticket_rows = []
    for row in all_rows:
        try:
            if row.is_displayed() and row.text.strip():
                ticket_rows.append(row)
        except:
            continue
    
    ticket_count = len(ticket_rows)
    logger.info(f"✅ Found {ticket_count} visible ticket rows in grid")
    
    # Compare with UI count if available
    if ui_count is not None:
        if ui_count != ticket_count:
            logger.warning(f"⚠️ Mismatch! UI shows {ui_count} tickets but grid has {ticket_count} rows")
            logger.warning(f"⚠️ Using UI count ({ui_count}) as the correct value")
            # Use only the first ui_count rows
            ticket_rows = ticket_rows[:ui_count]
            ticket_count = ui_count
    
    if ticket_count == 0:
        logger.warning("⚠️ No tickets found! Check if filters are correct.")
        logger.info("Tip: Try removing the date filter and run again to get all resolved tickets.")
        raise Exception("No tickets found with current filters")
    
    # ==================== STEP 8: EXTRACT TICKET DETAILS ====================
    logger.info("=" * 80)
    logger.info(f"STEP 8: Extracting details from {ticket_count} tickets")
    logger.info("=" * 80)
    
    tickets_data = []
    
    # Use index-based loop to avoid stale element issues
    for idx in range(1, ticket_count + 1):
        try:
            logger.info(f"Processing ticket {idx}/{ticket_count}")
            
            # Re-find ticket rows fresh each time to avoid stale elements
            ticket_rows = driver.find_elements(
                By.XPATH, "//div[contains(@class, 'x-grid3-body')]//div[contains(@class, 'x-grid3-row')]"
            )
            
            if idx > len(ticket_rows):
                logger.warning(f"  ⚠️ Ticket {idx} not found in grid (only {len(ticket_rows)} rows)")
                continue
            
            row = ticket_rows[idx - 1]  # Get the row for this index
            
            # Scroll row into view
            driver.execute_script("arguments[0].scrollIntoView(true);", row)
            time.sleep(0.5)
            
            # DOUBLE-CLICK row to open full details page
            logger.info(f"  Double-clicking ticket row {idx}...")
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(driver)
            actions.double_click(row).perform()
            time.sleep(4)  # Wait for details page to load
            
            # Switch to default content first
            driver.switch_to.default_content()
            time.sleep(2)  # Give more time for page to load
            
            # The details form is inside iframe id="ext-gen103" (x-managed-iframe)
            # We'll find it by looking for the Summary field (which we know exists)
            details_form_found = False
            try:
                iframes = driver.find_elements(By.XPATH, "//iframe[contains(@class, 'x-managed-iframe')]")
                logger.info(f"  Found {len(iframes)} managed iframes for details")
                for i, iframe in enumerate(iframes):
                    try:
                        driver.switch_to.frame(iframe)
                        # Check if this iframe has the Summary field (we know this works)
                        test = driver.find_element(By.XPATH, "//input[contains(@class, 'x-frs-id-Subject')]")
                        if test:
                            details_form_found = True
                            logger.info(f"  ✅ Found details form iframe (iframe {i+1})")
                            break
                    except:
                        driver.switch_to.default_content()
                        continue
                
                if not details_form_found:
                    driver.switch_to.default_content()
                    logger.warning(f"  ⚠️ Could not find details form iframe with Summary field")
            except Exception as e:
                logger.warning(f"  ⚠️ Error finding details iframe: {str(e)}")
                driver.switch_to.default_content()
            
            # Now extract Incident Number
            # Try multiple strategies since it might be in different locations
            incident_number = f"UNKNOWN_{idx}"
            import re
            
            # Strategy 1: If we found the details form iframe, try there first
            if details_form_found:
                try:
                    incident_header = driver.find_element(
                        By.XPATH, "//div[contains(@class, 'x-headerlabel') and contains(text(), 'Incident:')]"
                    )
                    header_text = incident_header.text
                    match = re.search(r'Incident:?\s*(\d+)', header_text)
                    if match:
                        incident_number = f"INC{match.group(1)}"
                        logger.info(f"  ✅ Incident Number (from form iframe): {incident_number}")
                except Exception as e:
                    logger.info(f"  Incident not in form iframe, trying other locations...")
            
            # Strategy 2: If not found, try in default content (page header)
            if incident_number.startswith("UNKNOWN"):
                try:
                    driver.switch_to.default_content()
                    incident_header = driver.find_element(
                        By.XPATH, "//div[contains(@class, 'x-headerlabel') and contains(text(), 'Incident:')]"
                    )
                    header_text = incident_header.text
                    match = re.search(r'Incident:?\s*(\d+)', header_text)
                    if match:
                        incident_number = f"INC{match.group(1)}"
                        logger.info(f"  ✅ Incident Number (from page header): {incident_number}")
                except Exception as e:
                    logger.info(f"  Incident not in page header, searching all iframes...")
            
            # Strategy 3: Search all iframes with relaxed XPath (any div containing "Incident:")
            if incident_number.startswith("UNKNOWN"):
                try:
                    driver.switch_to.default_content()
                    all_iframes = driver.find_elements(By.TAG_NAME, "iframe")
                    logger.info(f"  Searching {len(all_iframes)} iframes for Incident Number (relaxed search)...")
                    for i, iframe in enumerate(all_iframes):
                        try:
                            driver.switch_to.frame(iframe)
                            # Try multiple XPath strategies
                            xpaths = [
                                "//div[contains(text(), 'Incident:')]",
                                "//*[contains(text(), 'Incident:')]",
                                "//div[contains(@class, 'headerlabel')]",
                            ]
                            for xpath in xpaths:
                                try:
                                    incident_header = driver.find_element(By.XPATH, xpath)
                                    header_text = incident_header.text
                                    if 'Incident:' in header_text or 'Incident ' in header_text:
                                        match = re.search(r'Incident:?\s*(\d+)', header_text)
                                        if match:
                                            incident_number = f"INC{match.group(1)}"
                                            logger.info(f"  ✅ Incident Number (from iframe {i+1}, xpath strategy): {incident_number}")
                                            break
                                except:
                                    continue
                            if not incident_number.startswith("UNKNOWN"):
                                break
                        except:
                            pass
                        finally:
                            driver.switch_to.default_content()
                    
                    if incident_number.startswith("UNKNOWN"):
                        logger.warning(f"  ⚠️ Could not find Incident Number in any location with any strategy")
                except Exception as e:
                    logger.warning(f"  ⚠️ Error searching for Incident Number: {str(e)[:80]}")
            
            # Extract Summary, Description, and Owner
            summary = "N/A"
            description = "N/A"
            owner = "N/A"
            status = "Resolved"
            
            try:
                # Wait a bit more for the page to fully load
                time.sleep(1)
                
                # We should already be in the correct iframe from the Incident Number extraction
                # If not, try to find it
                try:
                    # Test if we can find the Summary field
                    test = driver.find_element(By.XPATH, "//input[contains(@class, 'x-frs-id-Subject')]")
                    logger.debug(f"  Already in correct iframe for form fields")
                except:
                    # Not in correct iframe, need to find it
                    logger.debug(f"  Not in form iframe, searching...")
                    driver.switch_to.default_content()
                    iframes = driver.find_elements(By.XPATH, "//iframe[contains(@class, 'x-managed-iframe')]")
                    for frame in iframes:
                        try:
                            driver.switch_to.frame(frame)
                            test = driver.find_element(By.XPATH, "//input[contains(@class, 'x-frs-id-Subject')]")
                            if test:
                                logger.debug(f"  Found form iframe")
                                break
                        except:
                            driver.switch_to.default_content()
                            continue
                
                # Find Summary field using the specific class identifier from HTML
                # The Summary field has class "x-frs-id-Subject" and frsqq_fname="Subject"
                summary_xpaths = [
                    # Strategy 1: Use the specific class identifier
                    "//input[contains(@class, 'x-frs-id-Subject')]",
                    # Strategy 2: Use the frsqq_fname attribute
                    "//input[@frsqq_fname='Subject']",
                    # Strategy 3: Combine both for more specificity
                    "//input[contains(@class, 'x-frs-id-Subject') and @frsqq_fname='Subject']",
                ]
                
                summary_element = None
                for idx, xpath in enumerate(summary_xpaths):
                    try:
                        summary_element = driver.find_element(By.XPATH, xpath)
                        if summary_element and summary_element.is_displayed():
                            logger.debug(f"  Found Summary using strategy {idx+1}")
                            break
                    except:
                        continue
                
                if summary_element:
                    summary = summary_element.get_attribute('value') or "N/A"
                    logger.info(f"  ✅ Summary: {summary[:100]}...")
                else:
                    logger.warning(f"  ⚠️ Could not find Summary field")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not extract Summary: {str(e)}")
            
            try:
                # Find Description field - it's in a rich text editor (iframe or div)
                # Look for the description content area
                description_element = driver.find_element(
                    By.XPATH, "//iframe[contains(@title, 'Description')] | //div[contains(@class, 'x-html-editor-wrap')]//iframe | //*[text()='Description']/following::iframe[1]"
                )
                # Switch to iframe if it's an iframe
                driver.switch_to.frame(description_element)
                description_body = driver.find_element(By.TAG_NAME, "body")
                description = description_body.text or "N/A"
                driver.switch_to.default_content()  # Switch back to main content
                logger.info(f"  ✅ Description: {description[:50]}...")
            except:
                # Try alternative: look for plain text description
                try:
                    description_element = driver.find_element(
                        By.XPATH, "//*[text()='Description']/following::div[contains(@class, 'x-form-item')]//div"
                    )
                    description = description_element.text or "N/A"
                    logger.info(f"  ✅ Description: {description[:50]}...")
                except:
                    logger.warning(f"  ⚠️ Could not extract Description")
            
            try:
                # IMPORTANT: After extracting Description, we did driver.switch_to.default_content()
                # We need to switch back into the form iframe to find Owner
                # Owner field is in the same iframe as Summary (x-managed-iframe / ext-gen103)
                
                # Switch back into the form iframe
                details_form_found_for_owner = False
                try:
                    iframes = driver.find_elements(By.XPATH, "//iframe[contains(@class, 'x-managed-iframe')]")
                    logger.debug(f"  Searching {len(iframes)} iframes for Owner field")
                    for i, frame in enumerate(iframes):
                        try:
                            driver.switch_to.frame(frame)
                            # Verify this is the form iframe by checking for Subject field
                            test = driver.find_element(By.XPATH, "//input[contains(@class, 'x-frs-id-Subject')]")
                            if test:
                                details_form_found_for_owner = True
                                logger.debug(f"  ✅ Re-entered form iframe {i+1} for Owner extraction")
                                break
                        except:
                            driver.switch_to.default_content()
                            continue
                    
                    if not details_form_found_for_owner:
                        logger.warning(f"  ⚠️ Could not re-enter form iframe for Owner")
                except Exception as e:
                    logger.warning(f"  ⚠️ Error re-entering form iframe: {str(e)}")
                
                # Owner field should now be accessible (if we're in the correct iframe)
                if details_form_found_for_owner:
                    # Find Owner field using the specific class identifier from HTML
                    # The Owner field has class "x-frs-id-Owner" and frsqq_fname="Owner"
                    
                    # First, let's check if we can find ANY input with Owner in class
                    try:
                        all_owner_inputs = driver.find_elements(By.XPATH, "//input[contains(@class, 'Owner') or contains(@frsqq_fname, 'Owner')]")
                        logger.info(f"  DEBUG: Found {len(all_owner_inputs)} inputs with 'Owner' in attributes")
                        for inp in all_owner_inputs[:3]:
                            try:
                                inp_class = inp.get_attribute('class')
                                inp_name = inp.get_attribute('frsqq_fname')
                                inp_value = inp.get_attribute('value')
                                inp_visible = inp.is_displayed()
                                logger.info(f"    - class='{inp_class}', frsqq_fname='{inp_name}', value='{inp_value}', visible={inp_visible}")
                            except:
                                pass
                    except Exception as e:
                        logger.warning(f"  DEBUG: Could not enumerate Owner inputs: {str(e)}")
                    
                    owner_xpaths = [
                        # Strategy 1: Explicitly exclude OwnerTeam
                        "//input[contains(@class, 'x-frs-id-Owner') and not(contains(@class, 'OwnerTeam'))]",
                        # Strategy 2: Use the frsqq_fname attribute
                        "//input[@frsqq_fname='Owner']",
                        # Strategy 3: Match with space boundaries
                        "//input[contains(concat(' ', @class, ' '), ' x-frs-id-Owner ')]",
                    ]
                    
                    owner_element = None
                    for idx, xpath in enumerate(owner_xpaths):
                        try:
                            elements = driver.find_elements(By.XPATH, xpath)
                            logger.info(f"  Owner strategy {idx+1}: Found {len(elements)} elements with XPath")
                            # Log what we found
                            for i, elem in enumerate(elements):
                                try:
                                    elem_class = elem.get_attribute('class')
                                    elem_value = elem.get_attribute('value')
                                    logger.info(f"    Element {i+1}: class contains '{elem_class.split('x-frs-id-')[1].split()[0] if 'x-frs-id-' in elem_class else 'unknown'}', value='{elem_value}'")
                                except:
                                    pass
                            
                            for elem in elements:
                                if elem.is_displayed():
                                    owner_element = elem
                                    logger.info(f"  ✅ Found visible Owner field using strategy {idx+1}")
                                    break
                            if owner_element:
                                break
                        except Exception as ex:
                            logger.warning(f"  Owner strategy {idx+1} failed: {str(ex)[:80]}")
                            continue
                    
                    if owner_element:
                        owner = owner_element.get_attribute('value') or "N/A"
                        logger.info(f"  ✅ Owner: {owner}")
                    else:
                        logger.warning(f"  ⚠️ Could not find Owner field with any strategy")
                else:
                    logger.warning(f"  ⚠️ Skipping Owner extraction - not in correct iframe")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not extract Owner: {str(e)}")
            
            # Store ticket data (Service field removed as not required)
            tickets_data.append({
                'Incident_Number': incident_number,
                'Summary': summary,
                'Description': description,
                'Owner': owner,
                'Status': status,
                'Resolved_Date': datetime.now().strftime('%Y-%m-%d'),
                'Extracted_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            logger.info(f"  ✅ Ticket {incident_number} extracted successfully")
            
            # Go back to list view
            logger.info(f"  Going back to list view...")
            
            # Switch back to default content first
            driver.switch_to.default_content()
            time.sleep(1)
            
            # The List View button is inside an iframe - try to find it
            # Search through all iframes to find the one with the List View button
            found_button_iframe = False
            try:
                all_iframes = driver.find_elements(By.TAG_NAME, "iframe")
                logger.info(f"  Searching through {len(all_iframes)} iframes for List View button...")
                
                for idx, iframe in enumerate(all_iframes):
                    try:
                        driver.switch_to.default_content()
                        driver.switch_to.frame(iframe)
                        
                        # Try to find the List View button in this iframe
                        try:
                            button = driver.find_element(By.XPATH, "//button[contains(@class, 'x-btn-text') and contains(text(), 'List View')]")
                            if button.is_displayed():
                                logger.info(f"  ✅ Found List View button in iframe {idx}")
                                found_button_iframe = True
                                break
                        except:
                            pass
                    except:
                        pass
                
                if not found_button_iframe:
                    driver.switch_to.default_content()
                    logger.warning(f"  ⚠️ Could not find List View button in any iframe")
            except Exception as e:
                logger.warning(f"  ⚠️ Error searching iframes: {str(e)}")
                driver.switch_to.default_content()
            
            # Try multiple strategies to go back
            back_clicked = False
            back_strategies = [
                # Strategy 1: Button with class x-btn-text and text "List View" (exact match from HTML)
                "//button[@class='x-btn-text' and text()='List View']",
                # Strategy 2: Button with class containing x-btn-text and text "List View"
                "//button[contains(@class, 'x-btn-text') and text()='List View']",
                # Strategy 3: Button with class x-btn-text containing "List View"
                "//button[contains(@class, 'x-btn-text') and contains(text(), 'List View')]",
                # Strategy 4: Any button with text "List View"
                "//button[text()='List View']",
                # Strategy 5: Button containing "List View" text
                "//button[contains(., 'List View')]",
                # Strategy 6: Look in toolbar for button with List View
                "//div[contains(@class, 'x-toolbar')]//button[contains(., 'List View')]",
                # Strategy 7: Look for any element with "List View" text that's clickable
                "//*[text()='List View' and @type='button']",
            ]
            
            for idx, xpath in enumerate(back_strategies):
                try:
                    # Try to find all matching elements
                    buttons = driver.find_elements(By.XPATH, xpath)
                    for button in buttons:
                        try:
                            if button.is_displayed() and button.is_enabled():
                                # Scroll button into view
                                driver.execute_script("arguments[0].scrollIntoView(true);", button)
                                time.sleep(0.3)
                                # Try clicking
                                button.click()
                                logger.info(f"  ✅ Clicked List View button using strategy {idx+1}")
                                back_clicked = True
                                break
                        except:
                            continue
                    if back_clicked:
                        break
                except:
                    continue
            
            if not back_clicked:
                logger.warning(f"  ⚠️ Could not find List View button with XPath, trying JavaScript...")
                
                # Try using JavaScript to find and click the button - with debug logging
                try:
                    # First, log ALL buttons to see what's available (including empty ones)
                    js_debug = """
                    var buttons = document.querySelectorAll('button');
                    var buttonInfo = [];
                    for (var i = 0; i < Math.min(buttons.length, 30); i++) {
                        var text = (buttons[i].textContent || buttons[i].innerText || '').trim();
                        var className = buttons[i].className || '';
                        var id = buttons[i].id || '';
                        // Log ALL buttons, even empty ones
                        buttonInfo.push({text: text, class: className, id: id});
                    }
                    return buttonInfo;
                    """
                    button_info = driver.execute_script(js_debug)
                    logger.info(f"  Found {len(button_info)} total buttons in current context:")
                    for i, info in enumerate(button_info[:15]):
                        logger.info(f"    [{i}] text='{info.get('text', '')}', class='{info.get('class', '')}', id='{info.get('id', '')}'")
                    
                    # Now try to click - be more flexible with text matching
                    js_script = """
                    var buttons = document.querySelectorAll('button');
                    for (var i = 0; i < buttons.length; i++) {
                        var text = (buttons[i].textContent || buttons[i].innerText || '').replace(/\\s+/g, ' ').trim();
                        var lowerText = text.toLowerCase();
                        if ((lowerText.indexOf('list') >= 0 && lowerText.indexOf('view') >= 0) || 
                            lowerText === 'list view') {
                            buttons[i].click();
                            return 'Clicked button with text: ' + text;
                        }
                    }
                    return false;
                    """
                    result = driver.execute_script(js_script)
                    if result and result != False:
                        logger.info(f"  ✅ {result}")
                        back_clicked = True
                    else:
                        logger.warning(f"  ⚠️ JavaScript could not find List View button")
                except Exception as e:
                    logger.warning(f"  ⚠️ JavaScript failed: {str(e)}")
            
            if not back_clicked:
                logger.warning(f"  ⚠️ All methods failed - stopping extraction to prevent cascading failures")
                break  # Exit the loop to prevent further failures
            
            time.sleep(3)
            
            # Switch back to the iframe with the grid
            try:
                # First switch to default content
                driver.switch_to.default_content()
                time.sleep(0.5)
                
                # Then switch to the grid iframe
                driver.switch_to.frame("ext-gen98")
                time.sleep(1)
                
                # Re-find ticket rows after going back
                ticket_rows = driver.find_elements(
                    By.XPATH, "//div[contains(@class, 'x-grid3-body')]//div[contains(@class, 'x-grid3-row')]"
                )
                logger.info(f"  ✅ Found {len(ticket_rows)} rows after going back")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not switch back to iframe: {str(e)}")
            
        except Exception as e:
            logger.error(f"  ❌ Error extracting ticket {idx}: {str(e)}")
            # Try to go back to list view if we're stuck
            try:
                driver.switch_to.default_content()
                time.sleep(1)
                
                # Try to find and click back button using improved strategies
                back_clicked = False
                back_xpaths = [
                    "//button[@class='x-btn-text' and text()='List View']",
                    "//button[contains(@class, 'x-btn-text') and text()='List View']",
                    "//button[contains(@class, 'x-btn-text') and contains(text(), 'List View')]",
                    "//button[text()='List View']",
                    "//button[contains(., 'List View')]",
                ]
                
                for xpath in back_xpaths:
                    try:
                        back_button = driver.find_element(By.XPATH, xpath)
                        if back_button.is_displayed() and back_button.is_enabled():
                            back_button.click()
                            back_clicked = True
                            logger.info(f"  Clicked List View in error handler")
                            break
                    except:
                        continue
                
                if not back_clicked:
                    # Try ESC key
                    from selenium.webdriver.common.keys import Keys
                    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
                    logger.info(f"  Pressed ESC key in error handler")
                
                time.sleep(2)
                # Switch back to iframe
                driver.switch_to.frame("ext-gen98")
                time.sleep(1)
                # Re-find rows
                ticket_rows = driver.find_elements(
                    By.XPATH, "//div[contains(@class, 'x-grid3-body')]//div[contains(@class, 'x-grid3-row')]"
                )
            except:
                logger.error(f"  ❌ Could not go back to list view")
                pass
            
            # Add placeholder data
            tickets_data.append({
                'Incident_Number': f"ERROR_{idx}",
                'Summary': "ERROR - Could not extract",
                'Description': "ERROR - Could not extract",
                'Owner': "N/A",
                'Service': "N/A",
                'Status': "Resolved",
                'Resolved_Date': datetime.now().strftime('%Y-%m-%d'),
                'Extracted_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            continue
    
    # ==================== STEP 9: SAVE TO EXCEL ====================
    logger.info("=" * 80)
    logger.info("STEP 9: Saving extracted data to Excel")
    logger.info("=" * 80)
    
    if len(tickets_data) == 0:
        logger.error("❌ No ticket data extracted!")
        raise Exception("No ticket data to save")
    
    # Create DataFrame
    df_tickets = pd.DataFrame(tickets_data)
    
    # Generate output filename
    output_file = os.path.join(
        OUTPUT_DIR,
        f"resolved_tickets_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.xlsx"
    )
    
    # Save to Excel
    df_tickets.to_excel(output_file, index=False)
    
    logger.info("=" * 80)
    logger.info("✅ EXTRACTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total tickets extracted: {len(tickets_data)}")
    logger.info(f"Successful extractions: {len([t for t in tickets_data if not t['Incident_Number'].startswith('ERROR')])}")
    logger.info(f"Failed extractions: {len([t for t in tickets_data if t['Incident_Number'].startswith('ERROR')])}")
    logger.info(f"Output file: {output_file}")
    logger.info("=" * 80)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total tickets: {len(tickets_data)}")
    print(f"Successful: {len([t for t in tickets_data if not t['Incident_Number'].startswith('ERROR')])}")
    print(f"Failed: {len([t for t in tickets_data if t['Incident_Number'].startswith('ERROR')])}")
    print(f"\nOutput file: {output_file}")
    print("=" * 80)
    
    # Show first few rows
    print("\nFirst 5 tickets:")
    print(df_tickets[['Incident_Number', 'Summary', 'Owner']].head())
    
except Exception as e:
    logger.error("=" * 80)
    logger.error(f"❌ FATAL ERROR: {str(e)}")
    logger.error("=" * 80)
    print(f"\n❌ Error: {str(e)}")
    raise

finally:
    # ==================== CLEANUP ====================
    logger.info("=" * 80)
    logger.info("Cleaning up and closing browser")
    logger.info("=" * 80)
    
    # Wait a moment before closing
    logger.info("Waiting 5 seconds before closing browser...")
    time.sleep(5)
    
    # Close browser
    driver.quit()
    logger.info("✅ Browser closed")
    logger.info("Script execution completed")
