# the main module to invoke other supportive modules
import cleaning_tools as ct

train_data, test_data = ct.read_data()
missing_count = check_missing_data(train_data)
print(missing_count)