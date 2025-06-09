import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox

def load_dataframe(parent):
    """
    Open a file dialog to select a CSV or Excel file and load it into a pandas DataFrame.
    """
    options = QFileDialog.Options()
    file_name, _ = QFileDialog.getOpenFileName(parent, "Load Data File", "",
                                               "All Files (*);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)",
                                               options=options)
    if not file_name:
        return None

    try:
        if file_name.endswith('.csv'):
            # TODO: Add options for separator, encoding, header row
            return pd.read_csv(file_name)
        elif file_name.endswith(('.xlsx', '.xls')):
            # TODO: Add option to select a sheet
            return pd.read_excel(file_name)
        else:
            # Inform the user that the file format is not supported
            raise ValueError("Unsupported file format")
    except Exception as e:
        # Show an error message to the user
        QMessageBox.critical(parent, "Error loading file", str(e))
        return None 