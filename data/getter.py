count = 0

for idx in range(1, 10_000_000):
    
    print(idx)
    
    url = f'https://rnaloops.cs.put.poznan.pl/search/details/{idx}'
    driver = webdriver.Chrome(options=get_chrome_options(headless=True))
    driver.get(url)
    
    for element in driver.find_elements(By.CLASS_NAME, 'MuiButtonBase-root'):
        
        if element.get_attribute('aria-label') == 'Download summary file':
            
            element.click()
            print(f'Downloaded file {count}')
            count += 1