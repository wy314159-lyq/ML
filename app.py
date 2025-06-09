import pandas as pd
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QTableView, QTabWidget,
                             QGroupBox, QLabel, QComboBox, QLineEdit,
                             QMessageBox, QListWidget, QListWidgetItem,
                             QAbstractItemView, QTextEdit, QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from data_handler import load_dataframe
from visualization import Plotter
from mpl_canvas import MplCanvas
from dialogs import ComparisonDialog
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
import webbrowser
import os
from ydata_profiling import ProfileReport
from category_encoders import TargetEncoder

# --- Worker for background tasks ---
class ProfileWorker(QObject):
    """
    A worker to generate the ydata-profiling report in a separate thread.
    """
    finished = pyqtSignal(str)  # Signal to emit the report file path when done
    error = pyqtSignal(str)     # Signal to emit error messages

    def __init__(self, df):
        super().__init__()
        self.df = df

    def run(self):
        try:
            profile = ProfileReport(self.df, title="Data Analysis Report")
            # Save to a temporary file
            report_path = os.path.join(os.path.dirname(__file__), "data_report.html")
            profile.to_file(report_path)
            self.finished.emit(report_path)
        except Exception as e:
            self.error.emit(str(e))

def set_dark_style(app):
    style_sheet = """
    QWidget{
        background-color: #2E2E2E;
        color: #F0F0F0;
        font-family: Arial;
        font-size: 10pt;
    }
    QMainWindow{
        background-color: #222222;
    }
    QTabWidget::pane {
        border-top: 2px solid #C2C7CB;
    }
    QTabBar::tab {
        background: #444444;
        border: 1px solid #2E2E2E;
        padding: 10px;
        min-width: 80px;
    }
    QTabBar::tab:selected, QTabBar::tab:hover {
        background: #555555;
    }
    QTabBar::tab:selected {
        border-color: #3D98D2;
        border-bottom-color: #3D98D2; 
    }
    QGroupBox {
        background-color: #3A3A3A;
        border: 1px solid gray;
        border-radius: 5px;
        margin-top: 1ex; /* leave space at the top for the title */
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center; /* position at the top center */
        padding: 0 3px;
    }
    QPushButton {
        background-color: #555555;
        border: 1px solid #3D98D2;
        padding: 5px;
        min-width: 70px;
        border-radius: 2px;
    }
    QPushButton:hover {
        background-color: #6A6A6A;
    }
    QPushButton:pressed {
        background-color: #3D98D2;
    }
    QTableView {
        background-color: #3A3A3A;
        gridline-color: #555555;
    }
    QHeaderView::section {
        background-color: #444444;
        padding: 4px;
        border: 1px solid #555555;
    }
    QComboBox {
        border: 1px solid gray;
        border-radius: 3px;
        padding: 1px 18px 1px 3px;
        min-width: 6em;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 1px;
        border-left-color: darkgray;
        border-left-style: solid;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
    }
    QLineEdit {
        background-color: #2E2E2E;
        padding: 2px;
        border: 1px solid gray;
        border-radius: 2px;
    }
    QTextEdit {
        background-color: #2E2E2E;
        border: 1px solid gray;
    }
    QListWidget {
        background-color: #3A3A3A;
    }
    """
    app.setStyleSheet(style_sheet)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Materials Science Data Explorer")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Data
        self.df = None
        self.X = None
        self.y = None
        self.operation_history = []
        self.thread = None # For managing background tasks

        self._init_ui()

    def _init_ui(self):
        # Left Panel: Data Import and Column Selection
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)

        # Data Import Group
        import_group = QGroupBox("Data Ingestion")
        import_layout = QVBoxLayout()
        self.btn_load_csv = QPushButton("Load CSV/Excel")
        self.btn_load_csv.clicked.connect(self.load_data)
        import_layout.addWidget(self.btn_load_csv)
        import_group.setLayout(import_layout)
        left_layout.addWidget(import_group)

        # Columns Group
        columns_group = QGroupBox("Columns")
        columns_layout = QVBoxLayout()
        self.column_list_widget = QListWidget()
        self.column_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.column_list_widget.itemSelectionChanged.connect(self.on_column_selection_changed)
        columns_layout.addWidget(self.column_list_widget)
        columns_group.setLayout(columns_layout)
        left_layout.addWidget(columns_group)

        self.main_layout.addWidget(left_panel)

        # Right Panel: Tabs for Data, EDA, Preprocessing
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)
        self.main_layout.addWidget(right_panel)

        # Tab 1: Data Preview
        self.tab_data_preview = QWidget()
        self.tabs.addTab(self.tab_data_preview, "Data Preview")
        preview_layout = QVBoxLayout(self.tab_data_preview)
        self.data_preview_table = QTableView()
        preview_layout.addWidget(self.data_preview_table)
        self.lbl_data_shape = QLabel("Data Shape: ")
        preview_layout.addWidget(self.lbl_data_shape)

        # Tab 2: EDA
        self.tab_eda = QWidget()
        self.tabs.addTab(self.tab_eda, "EDA")
        eda_layout = QVBoxLayout(self.tab_eda)

        # EDA -> Data Summary
        summary_group = QGroupBox("Data Summary")
        summary_layout = QVBoxLayout()
        
        # --- Profiling Report Button ---
        self.btn_generate_report = QPushButton("Generate Detailed Analysis Report")
        self.btn_generate_report.clicked.connect(self.generate_profile_report)
        summary_layout.addWidget(self.btn_generate_report)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.describe_table = QTableView()
        summary_layout.addWidget(QLabel("df.info():"))
        summary_layout.addWidget(self.info_text)
        summary_layout.addWidget(QLabel("df.describe():"))
        summary_layout.addWidget(self.describe_table)
        summary_group.setLayout(summary_layout)
        eda_layout.addWidget(summary_group)
        
        # EDA -> Data Quality Report
        quality_group = QGroupBox("Data Quality Report")
        quality_layout = QVBoxLayout()
        self.quality_table = QTableView()
        quality_layout.addWidget(self.quality_table)
        
        # Duplicate rows
        self.lbl_duplicate_rows = QLabel("Duplicate Rows: N/A")
        quality_layout.addWidget(self.lbl_duplicate_rows)
        self.btn_remove_duplicates = QPushButton("Remove Duplicates")
        self.btn_remove_duplicates.clicked.connect(self.remove_duplicates)
        quality_layout.addWidget(self.btn_remove_duplicates)

        quality_group.setLayout(quality_layout)
        eda_layout.addWidget(quality_group)
        
        # EDA -> Visualization
        vis_group = QGroupBox("Visualization")
        vis_layout = QVBoxLayout()
        
        # --- Single Variable Plot ---
        single_var_group = QGroupBox("Single Variable Plot")
        single_var_layout = QVBoxLayout()
        single_var_form_layout = QHBoxLayout()

        self.vis_column_selector = QComboBox()
        self.vis_plot_type_selector = QComboBox()
        self.vis_plot_type_selector.addItems(["Histogram", "Box Plot", "Bar Plot"])
        self.btn_generate_plot = QPushButton("Generate Plot")
        self.btn_generate_plot.clicked.connect(self.generate_plot)
        single_var_form_layout.addWidget(QLabel("Column:"))
        single_var_form_layout.addWidget(self.vis_column_selector)
        single_var_form_layout.addWidget(QLabel("Plot Type:"))
        single_var_form_layout.addWidget(self.vis_plot_type_selector)
        single_var_form_layout.addWidget(self.btn_generate_plot)
        single_var_layout.addLayout(single_var_form_layout)
        single_var_group.setLayout(single_var_layout)
        vis_layout.addWidget(single_var_group)

        # --- Multi Variable Plot ---
        multi_var_group = QGroupBox("Multi Variable Plot")
        multi_var_layout = QVBoxLayout()

        # Scatter Plot controls
        scatter_layout = QHBoxLayout()
        self.scatter_x_selector = QComboBox()
        self.scatter_y_selector = QComboBox()
        self.scatter_hue_selector = QComboBox()
        self.btn_generate_scatter = QPushButton("Generate Scatter Plot")
        self.btn_generate_scatter.clicked.connect(self.generate_scatter_plot)
        scatter_layout.addWidget(QLabel("X-Axis:"))
        scatter_layout.addWidget(self.scatter_x_selector)
        scatter_layout.addWidget(QLabel("Y-Axis:"))
        scatter_layout.addWidget(self.scatter_y_selector)
        scatter_layout.addWidget(QLabel("Color (Hue):"))
        scatter_layout.addWidget(self.scatter_hue_selector)
        scatter_layout.addWidget(self.btn_generate_scatter)
        multi_var_layout.addLayout(scatter_layout)

        # Heatmap button
        self.btn_generate_heatmap = QPushButton("Generate Correlation Heatmap")
        self.btn_generate_heatmap.clicked.connect(self.generate_correlation_heatmap)
        multi_var_layout.addWidget(self.btn_generate_heatmap)

        multi_var_group.setLayout(multi_var_layout)
        vis_layout.addWidget(multi_var_group)
        
        # Plotting Area
        self.plot_canvas = MplCanvas(self)
        # Add Navigation Toolbar
        self.toolbar = NavigationToolbar(self.plot_canvas, self)
        vis_layout.addWidget(self.toolbar)
        vis_layout.addWidget(self.plot_canvas)
        
        vis_group.setLayout(vis_layout)
        eda_layout.addWidget(vis_group)

        # Tab 3: Preprocessing
        self.tab_preprocessing = QWidget()
        self.tabs.addTab(self.tab_preprocessing, "Preprocessing")
        preprocess_layout = QVBoxLayout(self.tab_preprocessing)

        # --- Imputation Group ---
        imputation_group = QGroupBox("Imputation")
        imputation_layout = QVBoxLayout()

        # Method Selector
        imputation_form_layout = QHBoxLayout()
        self.imputation_method_selector = QComboBox()
        self.imputation_method_selector.addItems([
            "Remove Rows with Missing Values",
            "Fill with Mean",
            "Fill with Median",
            "Fill with Mode",
            "Fill with Constant",
            "KNN Imputation"
        ])
        imputation_form_layout.addWidget(QLabel("Method:"))
        imputation_form_layout.addWidget(self.imputation_method_selector)
        imputation_layout.addLayout(imputation_form_layout)

        # Constant Value Input (initially hidden)
        self.constant_input_layout = QHBoxLayout()
        self.constant_input_layout.addWidget(QLabel("Constant Value:"))
        self.imputation_constant_input = QLineEdit("0")
        self.constant_input_layout.addWidget(self.imputation_constant_input)
        imputation_layout.addLayout(self.constant_input_layout)
        self.constant_input_layout.itemAt(0).widget().hide()
        self.constant_input_layout.itemAt(1).widget().hide()

        # KNN K Value Input (initially hidden)
        self.knn_k_layout = QHBoxLayout()
        self.knn_k_layout.addWidget(QLabel("Neighbors (k):"))
        self.knn_k_spinbox = QSpinBox()
        self.knn_k_spinbox.setMinimum(1)
        self.knn_k_spinbox.setValue(5)
        self.knn_k_layout.addWidget(self.knn_k_spinbox)
        imputation_layout.addLayout(self.knn_k_layout)
        self.knn_k_layout.itemAt(0).widget().hide()
        self.knn_k_layout.itemAt(1).widget().hide()
        
        # Connect selector to show/hide inputs
        self.imputation_method_selector.currentTextChanged.connect(self.update_imputation_options)

        self.btn_apply_imputation = QPushButton("Apply Imputation")
        self.btn_apply_imputation.clicked.connect(self.apply_imputation)
        imputation_layout.addWidget(self.btn_apply_imputation)
        imputation_group.setLayout(imputation_layout)
        preprocess_layout.addWidget(imputation_group)

        # --- Data Type Conversion ---
        dtype_group = QGroupBox("Data Type Conversion")
        dtype_layout = QVBoxLayout()
        
        type_form_layout = QHBoxLayout()
        type_form_layout.addWidget(QLabel("Selected column will be converted. Choose one from left panel."))

        self.dtype_selector = QComboBox()
        self.dtype_selector.addItems(["int64", "float64", "category", "object"])
        
        self.btn_convert_dtype = QPushButton("Convert Type")
        self.btn_convert_dtype.clicked.connect(self.convert_dtype)
        
        type_form_layout.addWidget(QLabel("Convert to:"))
        type_form_layout.addWidget(self.dtype_selector)
        type_form_layout.addWidget(self.btn_convert_dtype)

        dtype_layout.addLayout(type_form_layout)
        dtype_group.setLayout(dtype_layout)
        preprocess_layout.addWidget(dtype_group)

        # --- Column Operations ---
        column_ops_group = QGroupBox("Column Operations")
        column_ops_layout = QVBoxLayout()
        self.btn_delete_columns = QPushButton("Delete Selected Columns")
        self.btn_delete_columns.clicked.connect(self.delete_columns)
        column_ops_layout.addWidget(self.btn_delete_columns)

        column_ops_layout.addWidget(QLabel("--- or ---")) # separator
        delete_thresh_form_layout = QHBoxLayout()
        delete_thresh_form_layout.addWidget(QLabel("Delete columns where missing values exceed:"))
        self.missing_thresh_spinbox = QDoubleSpinBox()
        self.missing_thresh_spinbox.setSuffix(" %")
        self.missing_thresh_spinbox.setRange(0.0, 100.0)
        self.missing_thresh_spinbox.setValue(50.0)
        delete_thresh_form_layout.addWidget(self.missing_thresh_spinbox)
        self.btn_delete_by_threshold = QPushButton("Delete by Threshold")
        self.btn_delete_by_threshold.clicked.connect(self.delete_by_threshold)
        delete_thresh_form_layout.addWidget(self.btn_delete_by_threshold)
        column_ops_layout.addLayout(delete_thresh_form_layout)

        column_ops_group.setLayout(column_ops_layout)
        preprocess_layout.addWidget(column_ops_group)

        # --- Outlier Handling ---
        outlier_group = QGroupBox("Outlier Handling")
        outlier_layout = QVBoxLayout()
        self.outlier_suggestion_label = QLabel("Suggestion: Select a numeric column to see recommendations.")
        self.outlier_suggestion_label.setStyleSheet("font-style: italic; color: #CCCCCC;")
        outlier_layout.addWidget(self.outlier_suggestion_label)
        outlier_layout.addWidget(QLabel("Select numeric columns from the list on the left to handle outliers."))
        
        outlier_controls_layout = QHBoxLayout()
        self.outlier_method_selector = QComboBox()
        self.outlier_method_selector.addItems(["IQR", "Z-score"])
        self.outlier_threshold_input = QLineEdit("1.5")
        self.outlier_threshold_input.setPlaceholderText("k for IQR, z for Z-score")
        self.outlier_handling_selector = QComboBox()
        self.outlier_handling_selector.addItems(["Remove Rows", "Cap/Winsorize"])
        self.btn_handle_outliers = QPushButton("Detect & Handle Outliers")
        
        outlier_controls_layout.addWidget(QLabel("Method:"))
        outlier_controls_layout.addWidget(self.outlier_method_selector)
        outlier_controls_layout.addWidget(QLabel("Threshold:"))
        outlier_controls_layout.addWidget(self.outlier_threshold_input)
        outlier_controls_layout.addWidget(QLabel("Action:"))
        outlier_controls_layout.addWidget(self.outlier_handling_selector)
        
        outlier_layout.addLayout(outlier_controls_layout)
        outlier_layout.addWidget(self.btn_handle_outliers)
        
        outlier_group.setLayout(outlier_layout)
        preprocess_layout.addWidget(outlier_group)

        # --- Feature Scaling ---
        scaling_group = QGroupBox("Feature Scaling")
        scaling_layout = QVBoxLayout()
        scaling_layout.addWidget(QLabel("Select numeric columns from the list on the left to scale."))
        scaling_form_layout = QHBoxLayout()
        scaling_form_layout.addWidget(QLabel("Method:"))
        self.scaling_method_selector = QComboBox()
        self.scaling_method_selector.addItems(["StandardScaler", "MinMaxScaler", "RobustScaler"])
        scaling_form_layout.addWidget(self.scaling_method_selector)
        self.btn_apply_scaling = QPushButton("Apply Scaling")
        self.btn_apply_scaling.clicked.connect(self.apply_scaling)
        scaling_form_layout.addWidget(self.btn_apply_scaling)
        scaling_layout.addLayout(scaling_form_layout)
        scaling_group.setLayout(scaling_layout)
        preprocess_layout.addWidget(scaling_group)

        # --- Categorical Encoding ---
        encoding_group = QGroupBox("Categorical Encoding")
        encoding_layout = QVBoxLayout()
        encoding_layout.addWidget(QLabel("Select categorical columns from the list on the left to encode."))
        encoding_form_layout = QHBoxLayout()
        encoding_form_layout.addWidget(QLabel("Method:"))
        self.encoding_method_selector = QComboBox()
        self.encoding_method_selector.addItems(["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"])
        encoding_form_layout.addWidget(self.encoding_method_selector)
        self.btn_apply_encoding = QPushButton("Apply Encoding")
        self.btn_apply_encoding.clicked.connect(self.apply_encoding)
        encoding_form_layout.addWidget(self.btn_apply_encoding)
        encoding_layout.addLayout(encoding_form_layout)
        encoding_group.setLayout(encoding_layout)
        preprocess_layout.addWidget(encoding_group)

        # --- Data Binning ---
        binning_group = QGroupBox("Data Binning (Discretization)")
        binning_layout = QVBoxLayout()
        binning_layout.addWidget(QLabel("Select numeric columns to convert to intervals. New columns will be created and originals removed."))
        
        binning_form_layout = QHBoxLayout()
        binning_form_layout.addWidget(QLabel("Number of Bins:"))
        self.n_bins_spinbox = QSpinBox()
        self.n_bins_spinbox.setMinimum(2)
        self.n_bins_spinbox.setValue(5)
        binning_form_layout.addWidget(self.n_bins_spinbox)
        
        binning_form_layout.addWidget(QLabel("Strategy:"))
        self.binning_strategy_selector = QComboBox()
        self.binning_strategy_selector.addItems(["Quantile (Equal Frequency)", "Uniform (Equal Width)", "KMeans"])
        binning_form_layout.addWidget(self.binning_strategy_selector)
        
        self.btn_apply_binning = QPushButton("Apply Binning")
        self.btn_apply_binning.clicked.connect(self.apply_binning)
        binning_form_layout.addWidget(self.btn_apply_binning)
        
        binning_layout.addLayout(binning_form_layout)
        binning_group.setLayout(binning_layout)
        preprocess_layout.addWidget(binning_group)

        # --- Pipeline Persistence ---
        pipeline_group = QGroupBox("Preprocessing Pipeline")
        pipeline_layout = QHBoxLayout()
        self.btn_save_pipeline = QPushButton("Save Pipeline")
        self.btn_load_pipeline = QPushButton("Load and Apply Pipeline")
        self.btn_save_pipeline.clicked.connect(self.save_pipeline)
        self.btn_load_pipeline.clicked.connect(self.load_and_apply_pipeline)
        pipeline_layout.addWidget(self.btn_save_pipeline)
        pipeline_layout.addWidget(self.btn_load_pipeline)
        pipeline_group.setLayout(pipeline_layout)
        preprocess_layout.addWidget(pipeline_group)

        # Tab 4: Feature/Target
        self.tab_feature_target = QWidget()
        self.tabs.addTab(self.tab_feature_target, "Feature & Target")
        ft_layout = QVBoxLayout(self.tab_feature_target)

        ft_main_layout = QHBoxLayout()

        # Available Columns
        available_group = QGroupBox("Available Columns")
        available_layout = QVBoxLayout()
        self.available_cols_list = QListWidget()
        self.available_cols_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        available_layout.addWidget(self.available_cols_list)
        available_group.setLayout(available_layout)

        # Controls
        controls_layout = QVBoxLayout()
        self.btn_add_feature = QPushButton(">>")
        self.btn_add_feature.setToolTip("Add to Features (X)")
        self.btn_remove_feature = QPushButton("<<")
        self.btn_remove_feature.setToolTip("Remove from Features (X)")
        self.btn_add_all = QPushButton("Add All >>")
        self.btn_remove_all = QPushButton("<< Remove All")
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_add_feature)
        controls_layout.addWidget(self.btn_remove_feature)
        controls_layout.addWidget(self.btn_add_all)
        controls_layout.addWidget(self.btn_remove_all)
        controls_layout.addStretch()

        # Selected Features
        features_group = QGroupBox("Selected Features (X)")
        features_layout = QVBoxLayout()
        self.features_list = QListWidget()
        self.features_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        features_layout.addWidget(self.features_list)
        features_group.setLayout(features_layout)

        ft_main_layout.addWidget(available_group)
        ft_main_layout.addLayout(controls_layout)
        ft_main_layout.addWidget(features_group)
        ft_layout.addLayout(ft_main_layout)

        # Target Selection
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Select Target (y):"))
        self.target_selector = QComboBox()
        target_layout.addWidget(self.target_selector)
        ft_layout.addLayout(target_layout)

        # Confirmation
        self.btn_confirm_selection = QPushButton("Confirm Feature and Target Selection")
        ft_layout.addWidget(self.btn_confirm_selection)
        
        self.lbl_selection_status = QLabel("Status: Awaiting selection.")
        ft_layout.addWidget(self.lbl_selection_status)

        # Data Output
        self.output_group = QGroupBox("Data Output")
        output_layout = QVBoxLayout()
        self.btn_export_data = QPushButton("Export Processed Data to CSV")
        self.btn_export_data.clicked.connect(self.export_processed_data)
        output_layout.addWidget(self.btn_export_data)
        self.output_group.setLayout(output_layout)
        ft_layout.addWidget(self.output_group)
        self.output_group.setVisible(False) # Initially hidden
        
        # Connect signals
        self.tabs.currentChanged.connect(self.update_feature_target_tab)
        self.btn_add_feature.clicked.connect(self.add_features)
        self.btn_remove_feature.clicked.connect(self.remove_features)
        self.btn_add_all.clicked.connect(self.add_all_features)
        self.btn_remove_all.clicked.connect(self.remove_all_features)
        self.btn_confirm_selection.clicked.connect(self.confirm_selection)
        self.btn_handle_outliers.clicked.connect(self.handle_outliers)

    def load_data(self):
        self.df = load_dataframe(self)
        if self.df is not None:
            self.update_data_preview()
            self.update_eda_info()
            self.update_vis_selectors()
            self.plotter = Plotter(self.df)

    def update_data_preview(self):
        if self.df is not None:
            model = QStandardItemModel(self.df.shape[0], self.df.shape[1])
            model.setHorizontalHeaderLabels(self.df.columns)
            for row in range(self.df.shape[0]):
                for col in range(self.df.shape[1]):
                    item = QStandardItem(str(self.df.iat[row, col]))
                    model.setItem(row, col, item)
            self.data_preview_table.setModel(model)
            self.lbl_data_shape.setText(f"Data Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            self.update_column_list()

    def update_column_list(self):
        self.column_list_widget.clear()
        if self.df is not None:
            for col in self.df.columns:
                self.column_list_widget.addItem(QListWidgetItem(col))

    def update_eda_info(self):
        if self.df is not None:
            # df.info()
            import io
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            self.info_text.setText(buffer.getvalue())

            # df.describe()
            desc_df = self.df.describe().T
            desc_df.insert(0, 'statistic', desc_df.index)
            desc_model = QStandardItemModel(desc_df.shape[0], desc_df.shape[1])
            desc_model.setHorizontalHeaderLabels(desc_df.columns)
            for row in range(desc_df.shape[0]):
                for col in range(desc_df.shape[1]):
                    item = QStandardItem(str(desc_df.iat[row, col]))
                    desc_model.setItem(row, col, item)
            self.describe_table.setModel(desc_model)
            
            # Data Quality
            self.update_quality_report()
    
    def update_vis_selectors(self):
        if self.df is not None:
            self.vis_column_selector.clear()
            self.vis_column_selector.addItems(self.df.columns)

            # Populate scatter plot selectors
            numeric_cols = self.df.select_dtypes(include='number').columns.tolist()
            all_cols = self.df.columns.tolist()
            
            self.scatter_x_selector.clear()
            self.scatter_x_selector.addItems(numeric_cols)
            self.scatter_y_selector.clear()
            self.scatter_y_selector.addItems(numeric_cols)
            self.scatter_hue_selector.clear()
            self.scatter_hue_selector.addItems(["None"] + all_cols)

    def generate_plot(self):
        if self.df is not None:
            selected_column = self.vis_column_selector.currentText()
            plot_type = self.vis_plot_type_selector.currentText()

            if not selected_column:
                QMessageBox.warning(self, "Warning", "Please select a column.")
                return

            if plot_type == "Histogram":
                self.plotter.plot_histogram(self.plot_canvas, selected_column)
            elif plot_type == "Box Plot":
                self.plotter.plot_boxplot(self.plot_canvas, selected_column)
            elif plot_type == "Bar Plot":
                self.plotter.plot_barplot(self.plot_canvas, selected_column)

    def generate_scatter_plot(self):
        if self.df is not None:
            x_col = self.scatter_x_selector.currentText()
            y_col = self.scatter_y_selector.currentText()
            hue_col = self.scatter_hue_selector.currentText()

            if not x_col or not y_col:
                QMessageBox.warning(self, "Warning", "Please select columns for both X and Y axes.")
                return

            if hue_col == "None":
                hue_col = None

            self.plotter.plot_scatter(self.plot_canvas, x_col, y_col, hue_col)

    def generate_correlation_heatmap(self):
        if self.df is not None:
            self.plotter.plot_correlation_heatmap(self.plot_canvas)

    def update_quality_report(self):
        if self.df is not None:
            # Missing values and unique values
            missing_values = self.df.isnull().sum()
            missing_percent = (missing_values / len(self.df) * 100).round(2)
            unique_values = self.df.nunique()
            
            quality_df = pd.DataFrame({
                'Missing Values': missing_values,
                'Missing (%)': missing_percent,
                'Unique Values': unique_values
            })
            quality_df.insert(0, 'Column', quality_df.index)

            quality_model = QStandardItemModel(quality_df.shape[0], quality_df.shape[1])
            quality_model.setHorizontalHeaderLabels(quality_df.columns)
            for row in range(quality_df.shape[0]):
                for col in range(quality_df.shape[1]):
                    item = QStandardItem(str(quality_df.iat[row, col]))
                    quality_model.setItem(row, col, item)
            self.quality_table.setModel(quality_model)

            # Duplicate rows
            num_duplicates = self.df.duplicated().sum()
            self.lbl_duplicate_rows.setText(f"Duplicate Rows: {num_duplicates}")

    def remove_duplicates(self):
        if self.df is not None:
            reply = QMessageBox.question(self, 'Confirm Deletion', 
                                         f"Are you sure you want to remove {self.df.duplicated().sum()} duplicate rows?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                if self.df.duplicated().sum() > 0:
                    # Log operation before changing the dataframe
                    self.operation_history.append({'name': 'remove_duplicates'})
                    self.df = self.df.drop_duplicates()
                    self.df = self.df.reset_index(drop=True)
                    QMessageBox.information(self, "Success", "Duplicate rows removed.")
                    # Refresh views
                    self.update_data_preview()
                    self.update_eda_info()
            else:
                QMessageBox.information(self, "Info", "No duplicate rows found.")

    def update_imputation_options(self, text):
        # Hide all optional inputs first
        self.constant_input_layout.itemAt(0).widget().hide()
        self.constant_input_layout.itemAt(1).widget().hide()
        self.knn_k_layout.itemAt(0).widget().hide()
        self.knn_k_layout.itemAt(1).widget().hide()

        # Show inputs based on selected method
        if text == "Fill with Constant":
            self.constant_input_layout.itemAt(0).widget().show()
            self.constant_input_layout.itemAt(1).widget().show()
        elif text == "KNN Imputation":
            self.knn_k_layout.itemAt(0).widget().show()
            self.knn_k_layout.itemAt(1).widget().show()

    def apply_imputation(self):
        if self.df is None:
            return

        selected_items = self.column_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select one or more columns from the list on the left.")
            return

        selected_columns = [item.text() for item in selected_items]
        method = self.imputation_method_selector.currentText()

        for col in selected_columns:
            if method == "Remove Rows with Missing Values":
                self.df = self.df.dropna(subset=[col])
            elif method == "Fill with Mean":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    mean_val = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(mean_val)
                else:
                    QMessageBox.warning(self, "Warning", f"Column '{col}' is not numeric. Cannot fill with mean.")
            elif method == "Fill with Median":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                else:
                    QMessageBox.warning(self, "Warning", f"Column '{col}' is not numeric. Cannot fill with median.")
            elif method == "Fill with Mode":
                mode_val = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_val)
            elif method == "Fill with Constant":
                constant_val = self.imputation_constant_input.text()
                try:
                    # Try to convert to the column's type
                    dtype = self.df[col].dtype
                    self.df[col] = self.df[col].fillna(dtype.type(constant_val))
                except (ValueError, TypeError):
                     self.df[col] = self.df[col].fillna(constant_val)
        
        if method == "KNN Imputation":
            try:
                from sklearn.impute import KNNImputer
                import numpy as np
            except ImportError:
                QMessageBox.critical(self, "Error", "Scikit-learn is required for KNN Imputation. Please install it (`pip install scikit-learn`).")
                return

            numeric_cols = self.df[selected_columns].select_dtypes(include=np.number).columns.tolist()
            non_numeric_cols = list(set(selected_columns) - set(numeric_cols))

            if not numeric_cols:
                QMessageBox.warning(self, "Warning", "KNN Imputation can only be applied to numeric columns. No numeric columns were selected.")
                return
            
            if non_numeric_cols:
                QMessageBox.warning(self, "Info", f"KNN Imputation will only be applied to the following numeric columns:\n\n{', '.join(numeric_cols)}")

            k_value = self.knn_k_spinbox.value()
            imputer = KNNImputer(n_neighbors=k_value)
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])

        # Log operation
        op_details = {'name': 'apply_imputation', 'columns': selected_columns, 'method': method}
        if method == "Fill with Constant":
            op_details['constant_value'] = self.imputation_constant_input.text()
        if method == "KNN Imputation":
            op_details['k_value'] = self.knn_k_spinbox.value()
        self.operation_history.append(op_details)

        self.df = self.df.reset_index(drop=True)
        QMessageBox.information(self, "Success", "Imputation applied.")
        # Refresh views
        self.update_data_preview()
        self.update_eda_info()

    def convert_dtype(self):
        if self.df is None:
            return

        selected_items = self.column_list_widget.selectedItems()
        if len(selected_items) != 1:
            QMessageBox.warning(self, "Warning", "Please select exactly one column from the list on the left.")
            return

        column_name = selected_items[0].text()
        target_type = self.dtype_selector.currentText()

        try:
            self.df[column_name] = self.df[column_name].astype(target_type)
            # Log operation
            self.operation_history.append({'name': 'convert_dtype', 'column': column_name, 'target_type': target_type})
            QMessageBox.information(self, "Success", f"Column '{column_name}' converted to {target_type}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not convert column '{column_name}'.\nError: {e}")

        # Refresh views
        self.update_data_preview()
        self.update_eda_info()

    def delete_columns(self):
        if self.df is None:
            return

        selected_items = self.column_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select one or more columns from the list on the left to delete.")
            return

        selected_columns = [item.text() for item in selected_items]
        
        reply = QMessageBox.question(self, 'Confirm Deletion', 
                                     f"Are you sure you want to delete the following columns?\n\n{', '.join(selected_columns)}",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.df = self.df.drop(columns=selected_columns)
            # Log operation
            self.operation_history.append({'name': 'delete_columns', 'columns': selected_columns})
            QMessageBox.information(self, "Success", "Selected columns have been deleted.")
            # Refresh all views
            self.update_data_preview()
            self.update_eda_info()
            self.update_vis_selectors() 

    def update_feature_target_tab(self, index):
        if self.tabs.tabText(index) == "Feature & Target" and self.df is not None:
            self.available_cols_list.clear()
            self.features_list.clear()
            self.target_selector.clear()
            
            all_cols = self.df.columns.tolist()
            self.available_cols_list.addItems(all_cols)
            self.target_selector.addItems(["Select a Target"] + all_cols)

    def add_features(self):
        selected_items = self.available_cols_list.selectedItems()
        for item in selected_items:
            # Move item
            self.features_list.addItem(item.text())
            self.available_cols_list.takeItem(self.available_cols_list.row(item))
            
    def remove_features(self):
        selected_items = self.features_list.selectedItems()
        for item in selected_items:
            # Move item
            self.available_cols_list.addItem(item.text())
            self.features_list.takeItem(self.features_list.row(item))
            
    def add_all_features(self):
        while self.available_cols_list.count() > 0:
            item = self.available_cols_list.takeItem(0)
            self.features_list.addItem(item)
            
    def remove_all_features(self):
        while self.features_list.count() > 0:
            item = self.features_list.takeItem(0)
            self.available_cols_list.addItem(item)
            
    def confirm_selection(self):
        if self.df is None: return

        # Get features
        feature_names = []
        for i in range(self.features_list.count()):
            feature_names.append(self.features_list.item(i).text())

        # Get target
        target_name = self.target_selector.currentText()

        # Validation
        if not feature_names:
            QMessageBox.warning(self, "Error", "Please select at least one feature (X).")
            return
        if target_name == "Select a Target":
            QMessageBox.warning(self, "Error", "Please select a target (y).")
            return
        if target_name in feature_names:
            QMessageBox.warning(self, "Error", "Target (y) cannot also be a feature (X).")
            return
            
        self.X = self.df[feature_names]
        self.y = self.df[target_name]
        
        status_text = (f"Confirmed! Features (X): {self.X.shape[1]} columns. Target (y): '{target_name}'. "
                       f"Data shape: {self.X.shape[0]} samples.")
        self.lbl_selection_status.setText(status_text)
        QMessageBox.information(self, "Success", "Feature and Target selection confirmed.")
        self.output_group.setVisible(True) # Show the export option

    def save_pipeline(self):
        import json
        if not self.operation_history:
            QMessageBox.warning(self, "Warning", "No preprocessing steps have been performed yet.")
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Pipeline File", "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if not file_name:
            return

        try:
            with open(file_name, 'w') as f:
                json.dump(self.operation_history, f, indent=4)
            QMessageBox.information(self, "Success", f"Pipeline saved to:\n{file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save pipeline.\nError: {e}")

    def load_and_apply_pipeline(self):
        import json
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load data before applying a pipeline.")
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Pipeline File", "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if not file_name:
            return

        try:
            with open(file_name, 'r') as f:
                pipeline_operations = json.load(f)

            # Ask for confirmation
            reply = QMessageBox.question(self, 'Confirm Pipeline Application',
                                         f"This will apply {len(pipeline_operations)} preprocessing steps to your current data. This action cannot be undone. Proceed?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
            
            # Reset history before applying new one
            self.operation_history = []
            
            # Apply operations
            for op in pipeline_operations:
                self._execute_operation(op)

            QMessageBox.information(self, "Success", "Pipeline applied successfully.")
            # Refresh all views
            self.update_data_preview()
            self.update_eda_info()
            self.update_vis_selectors()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply pipeline.\nError: {e}")

    def _execute_operation(self, operation):
        """A dispatcher to run operations from a pipeline file."""
        op_name = operation.get('name')
        
        if op_name == 'remove_duplicates':
            self.df = self.df.drop_duplicates()
            self.df = self.df.reset_index(drop=True)
        
        elif op_name == 'apply_imputation':
            columns = operation['columns']
            method = operation['method']
            for col in columns:
                if method == "Remove Rows with Missing Values":
                    self.df = self.df.dropna(subset=[col])
                elif method == "Fill with Mean":
                    mean_val = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(mean_val)
                elif method == "Fill with Median":
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                elif method == "Fill with Mode":
                    mode_val = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(mode_val)
                elif method == "Fill with Constant":
                    constant_val = operation['constant_value']
                    dtype = self.df[col].dtype
                    self.df[col] = self.df[col].fillna(dtype.type(constant_val))
                elif method == "KNN Imputation":
                    from sklearn.impute import KNNImputer
                    import numpy as np
                    numeric_cols_in_op = self.df[columns].select_dtypes(include=np.number).columns.tolist()
                    # Use k from history, or default if not present (for backward compatibility)
                    k_value = operation.get('k_value', 5)
                    imputer = KNNImputer(n_neighbors=k_value)
                    self.df[numeric_cols_in_op] = imputer.fit_transform(self.df[numeric_cols_in_op])
                    break # Avoid iterating over columns for KNN

        elif op_name == 'convert_dtype':
            self.df[operation['column']] = self.df[operation['column']].astype(operation['target_type'])
            
        elif op_name == 'delete_by_threshold':
            threshold = operation['threshold']
            missing_percent = self.df.isnull().sum() / len(self.df) * 100
            cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
            if cols_to_drop:
                # Log operation before changing the dataframe
                self.operation_history.append({'name': 'delete_by_threshold', 'threshold': threshold})
                self.df = self.df.drop(columns=cols_to_drop)
                
        elif op_name == 'delete_columns':
            # Log operation before changing the dataframe
            self.operation_history.append({'name': 'delete_columns', 'columns': operation['columns']})
            self.df = self.df.drop(columns=operation['columns'])
        
        elif op_name == 'apply_scaling':
            columns = operation.get('columns', [])
            method = operation.get('method')
            if not method or not columns: return
            
            scaler_map = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}
            scaler = scaler_map.get(method)
            if not scaler: return

            valid_columns = [col for col in columns if col in self.df.columns]
            if valid_columns:
                self.df[valid_columns] = scaler.fit_transform(self.df[valid_columns])

        elif op_name == 'apply_encoding':
            columns = operation.get('columns', [])
            method = operation.get('method')
            if not columns or not method: return

            valid_columns = [col for col in columns if col in self.df.columns]
            if not valid_columns: return

            if method == "OneHotEncoder":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(self.df[valid_columns])
                new_col_names = encoder.get_feature_names_out(valid_columns)
                encoded_df = pd.DataFrame(encoded_data, columns=new_col_names, index=self.df.index)
                self.df = self.df.drop(columns=valid_columns)
                self.df = pd.concat([self.df, encoded_df], axis=1)

            elif method == "OrdinalEncoder":
                encoder = OrdinalEncoder()
                self.df[valid_columns] = encoder.fit_transform(self.df[valid_columns])

            elif method == "TargetEncoder":
                if self.y is None:
                    print("Warning: Target column not set. Skipping TargetEncoder during pipeline execution.")
                    return
                
                target_name = self.y.name
                if target_name not in self.df.columns:
                    print(f"Warning: Target column '{target_name}' not found. Skipping TargetEncoder step.")
                    return

                y_series = self.df[target_name]
                X_cols = [col for col in valid_columns if col != target_name]
                if not X_cols:
                    return
                
                encoder = TargetEncoder(cols=X_cols)
                self.df[X_cols] = encoder.fit_transform(self.df[X_cols], y_series)

        elif op_name == 'apply_binning':
            columns = operation.get('columns', [])
            n_bins = operation.get('n_bins', 5)
            strategy = operation.get('strategy', 'quantile')

            valid_columns = [col for col in columns if col in self.df.columns]
            if not valid_columns:
                return

            for col in valid_columns:
                try:
                    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                    # KBinsDiscretizer expects 2D array, so we reshape
                    binned_data = discretizer.fit_transform(self.df[[col]])
                    
                    # Create a new column for the binned data
                    new_col_name = f"{col}_binned"
                    self.df[new_col_name] = binned_data.astype(int)
                
                except Exception:
                    # If binning fails for one column, skip it and continue
                    continue
            
            self.df = self.df.drop(columns=valid_columns, errors='ignore')

    def export_processed_data(self):
        if self.X is None or self.y is None:
            QMessageBox.warning(self, "Error", "Please confirm your Feature (X) and Target (y) selection first.")
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Processed Data", "",
                                                   "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if not file_name:
            return

        try:
            # Combine features and target for export
            processed_df = pd.concat([self.X, self.y], axis=1)
            processed_df.to_csv(file_name, index=False)
            QMessageBox.information(self, "Success", f"Data successfully exported to:\n{file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data.\nError: {e}") 

    def delete_by_threshold(self):
        if self.df is None:
            return

        threshold = self.missing_thresh_spinbox.value()
        
        missing_percent = self.df.isnull().sum() / len(self.df) * 100
        cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()

        if not cols_to_drop:
            QMessageBox.information(self, "Info", "No columns exceed the specified missing value threshold.")
            return

        reply = QMessageBox.question(self, 'Confirm Deletion',
                                     f"The following columns exceed {threshold}% missing values and will be deleted:\n\n{', '.join(cols_to_drop)}\n\nProceed?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.df = self.df.drop(columns=cols_to_drop)
            # Log operation
            self.operation_history.append({'name': 'delete_by_threshold', 'threshold': threshold})
            QMessageBox.information(self, "Success", "Columns have been deleted.")
            # Refresh all views
            self.update_data_preview()
            self.update_eda_info()
            self.update_vis_selectors() 

    def handle_outliers(self):
        if self.df is None:
            return

        selected_items = self.column_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select one or more columns from the list on the left.")
            return

        selected_columns = [item.text() for item in selected_items]
        numeric_cols = self.df[selected_columns].select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            QMessageBox.warning(self, "Warning", "Outlier handling can only be applied to numeric columns. Please select numeric columns.")
            return
        
        if len(numeric_cols) < len(selected_columns):
             QMessageBox.information(self, "Info", f"Outlier handling will only be applied to the following numeric columns:\n\n{', '.join(numeric_cols)}")


        method = self.outlier_method_selector.currentText()
        action = self.outlier_handling_selector.currentText()
        try:
            threshold = float(self.outlier_threshold_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid number for the threshold.")
            return

        total_outlier_indices = set()
        df_modified = self.df.copy()
        
        for col in numeric_cols:
            if method == "IQR":
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
            else: # Z-score
                mean = self.df[col].mean()
                std = self.df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            
            if action == "Remove Rows":
                total_outlier_indices.update(self.df[outlier_mask].index)
            else: # Cap/Winsorize
                # Create a temporary series with capped values to show in plot
                temp_capped_series = df_modified[col].copy()
                temp_capped_series.loc[self.df[col] < lower_bound] = lower_bound
                temp_capped_series.loc[self.df[col] > upper_bound] = upper_bound

                # Check if there are any changes to show
                if not self.df[col].equals(temp_capped_series):
                    plotter = Plotter(self.df)
                    # Show comparison and get user confirmation
                    is_accepted = plotter.plot_comparison(self.df[col], temp_capped_series, col)
                    
                    if is_accepted:
                        # If user clicks OK, apply the change to the main modified dataframe
                        df_modified[col] = temp_capped_series
                    # If user clicks Cancel, do nothing, changes to this col are discarded
                else:
                    # Inform user that no outliers were found for this specific column
                    QMessageBox.information(self, "Info", f"No outliers detected in column '{col}' with the current settings.")

        # --- Apply the changes ---
        if action == "Remove Rows":
            if not total_outlier_indices:
                QMessageBox.information(self, "Info", "No outliers found to remove.")
                return
            reply = QMessageBox.question(self, 'Confirm Action', 
                                         f"Found {len(total_outlier_indices)} rows containing outliers. Remove them?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.df = self.df.drop(index=list(total_outlier_indices))
                self.operation_history.append({'name': 'handle_outliers', 'columns': numeric_cols, 'method': method, 'threshold': threshold, 'action': action})
                QMessageBox.information(self, "Success", f"{len(total_outlier_indices)} rows removed.")
        else: # Cap/Winsorize
            # Check if the dataframe was actually modified
            if self.df.equals(df_modified):
                QMessageBox.information(self, "Info", "No outliers were capped (or all changes were canceled).")
                return

            num_changed_cells = (self.df[numeric_cols] != df_modified[numeric_cols]).sum().sum()
            self.df = df_modified
            self.operation_history.append({'name': 'handle_outliers', 'columns': numeric_cols, 'method': method, 'threshold': threshold, 'action': action})
            QMessageBox.information(self, "Success", f"Capping applied to {num_changed_cells} outlier cells.")

        # Refresh all views
        self.df = self.df.reset_index(drop=True)
        self.update_data_preview()
        self.update_eda_info()
        self.update_vis_selectors() 

    def apply_encoding(self):
        if self.df is None:
            return

        selected_items = self.column_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select one or more columns for encoding.")
            return

        selected_columns = [item.text() for item in selected_items]
        method = self.encoding_method_selector.currentText()

        if method == "OneHotEncoder":
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(self.df[selected_columns])
            new_col_names = encoder.get_feature_names_out(selected_columns)
            encoded_df = pd.DataFrame(encoded_data, columns=new_col_names, index=self.df.index)
            self.df = self.df.drop(columns=selected_columns)
            self.df = pd.concat([self.df, encoded_df], axis=1)
            QMessageBox.information(self, "Success", f"OneHotEncoder applied. Original columns removed and {len(new_col_names)} new columns added.")

        elif method == "OrdinalEncoder":
            encoder = OrdinalEncoder()
            self.df[selected_columns] = encoder.fit_transform(self.df[selected_columns])
            QMessageBox.information(self, "Success", "OrdinalEncoder applied to selected columns.")

        elif method == "TargetEncoder":
            if self.y is None:
                QMessageBox.critical(self, "Error", "Target variable (y) is not set.\nPlease go to the 'Feature & Target' tab to select a target before using TargetEncoder.")
                return
            
            target_name = self.y.name
            if target_name in selected_columns:
                QMessageBox.warning(self, "Warning", f"The target column '{target_name}' cannot be used as a feature for Target Encoding. It will be ignored.")
                selected_columns.remove(target_name)
                if not selected_columns:
                    return

            encoder = TargetEncoder(cols=selected_columns)
            self.df[selected_columns] = encoder.fit_transform(self.df[selected_columns], self.y)
            QMessageBox.information(self, "Success", "TargetEncoder applied to selected columns.")

        self.operation_history.append({
            'name': 'apply_encoding',
            'columns': selected_columns,
            'method': method
        })
        
        self.update_data_preview()
        self.update_eda_info()
        self.update_vis_selectors()
        self.update_column_list()

    def apply_binning(self):
        if self.df is None:
            return

        selected_items = self.column_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select one or more numeric columns for binning.")
            return

        selected_columns = [item.text() for item in selected_items]
        
        # Filter for numeric columns
        numeric_cols = self.df[selected_columns].select_dtypes(include=np.number).columns.tolist()
        non_numeric_cols = list(set(selected_columns) - set(numeric_cols))

        if not numeric_cols:
            QMessageBox.warning(self, "Warning", "Binning can only be applied to numeric columns.")
            return
        
        if non_numeric_cols:
            QMessageBox.information(self, "Info", f"Binning will only be applied to the following numeric columns:\n\n{', '.join(numeric_cols)}")

        n_bins = self.n_bins_spinbox.value()
        strategy_map = {
            "Quantile (Equal Frequency)": "quantile",
            "Uniform (Equal Width)": "uniform",
            "KMeans": "kmeans"
        }
        strategy = strategy_map[self.binning_strategy_selector.currentText()]

        try:
            # We will create new columns and drop the old ones
            for col in numeric_cols:
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                # KBinsDiscretizer expects 2D array, so we reshape
                binned_data = discretizer.fit_transform(self.df[[col]])
                
                # Create a new column for the binned data
                new_col_name = f"{col}_binned"
                self.df[new_col_name] = binned_data.astype(int)
            
            # Drop original columns after processing all selected ones
            self.df = self.df.drop(columns=numeric_cols)

            # Log operation
            self.operation_history.append({
                'name': 'apply_binning',
                'columns': numeric_cols,
                'n_bins': n_bins,
                'strategy': strategy
            })

            QMessageBox.information(self, "Success", f"Binning applied to {len(numeric_cols)} columns. Original columns removed.")
            
            # Refresh all views
            self.update_data_preview()
            self.update_eda_info()
            self.update_vis_selectors()
            self.update_column_list()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply binning.\nError: {e}") 

    def generate_profile_report(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return

        # Disable the button to prevent multiple clicks
        self.btn_generate_report.setEnabled(False)
        self.btn_generate_report.setText("Generating Report...")

        # Create worker and thread
        self.thread = QThread()
        self.worker = ProfileWorker(self.df.copy()) # Use a copy to avoid thread issues
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_report_finished)
        self.worker.error.connect(self.on_report_error)
        
        # Start the thread
        self.thread.start()

    def on_report_finished(self, report_path):
        """Called when the report is successfully generated."""
        QMessageBox.information(self, "Success", f"Report generated! Opening in your browser...")
        webbrowser.open(f'file:///{os.path.realpath(report_path)}')
        
        # Clean up the thread
        self.thread.quit()
        self.thread.wait()
        self.btn_generate_report.setEnabled(True)
        self.btn_generate_report.setText("Generate Detailed Analysis Report")

    def on_report_error(self, error_message):
        """Called when report generation fails."""
        QMessageBox.critical(self, "Error", f"Failed to generate report.\n\nError: {error_message}")
        
        # Clean up the thread
        self.thread.quit()
        self.thread.wait()
        self.btn_generate_report.setEnabled(True)
        self.btn_generate_report.setText("Generate Detailed Analysis Report") 

    def apply_scaling(self):
        if self.df is None:
            return

        selected_items = self.column_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select one or more columns from the list on the left.")
            return

        selected_columns = [item.text() for item in selected_items]
        method = self.scaling_method_selector.currentText()

        numeric_cols = self.df[selected_columns].select_dtypes(include=np.number).columns.tolist()
        non_numeric_cols = list(set(selected_columns) - set(numeric_cols))

        if not numeric_cols:
            QMessageBox.warning(self, "Warning", "Scaling can only be applied to numeric columns. No numeric columns were selected.")
            return
        
        if non_numeric_cols:
            QMessageBox.information(self, "Info", f"Scaling will only be applied to the following numeric columns:\n\n{', '.join(numeric_cols)}")

        scaler_map = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler
        }
        scaler = scaler_map[method]()
        
        df_modified = self.df.copy()
        plotter = Plotter(self.df)
        changed_cols = []

        for col in numeric_cols:
            transformed_series = pd.Series(
                scaler.fit_transform(self.df[[col]]).flatten(), 
                index=self.df.index,
                name=col
            )
            
            is_accepted = plotter.plot_comparison(self.df[col], transformed_series, col)
            if is_accepted:
                df_modified[col] = transformed_series
                changed_cols.append(col)

        if not changed_cols:
            QMessageBox.information(self, "Info", "No scaling changes were applied (or all were canceled).")
            return
            
        self.df = df_modified
        
        self.operation_history.append({
            'name': 'apply_scaling',
            'columns': changed_cols,
            'method': method
        })

        QMessageBox.information(self, "Success", f"{method} applied to {len(changed_cols)} columns.")
        
        self.update_data_preview()
        self.update_eda_info() 

    def on_column_selection_changed(self):
        """Called when the user changes the column selection in the QListWidget."""
        if self.df is None:
            self.outlier_suggestion_label.setText("Suggestion: Select a numeric column to see recommendations.")
            return

        selected_items = self.column_list_widget.selectedItems()
        
        # We only provide suggestions for single-column selections for simplicity
        if len(selected_items) == 1:
            column_name = selected_items[0].text()
            
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(self.df[column_name]):
                # Calculate skewness
                skewness = self.df[column_name].skew()
                
                # Provide recommendation based on skewness
                if abs(skewness) > 1:
                    self.outlier_suggestion_label.setText(f"Suggestion: Skewness is {skewness:.2f}. Consider using IQR method as it's robust to skewed data.")
                    self.outlier_suggestion_label.setVisible(True)
                else:
                    self.outlier_suggestion_label.setText(f"Suggestion: Skewness is {skewness:.2f}. Both methods can be suitable.")
                    self.outlier_suggestion_label.setVisible(True)
            else:
                self.outlier_suggestion_label.setText("Suggestion: Selected column is not numeric.")
                self.outlier_suggestion_label.setVisible(True)
        else:
            # Hide or reset suggestion if multiple or no columns are selected
            self.outlier_suggestion_label.setText("Suggestion: Select a single numeric column to see recommendations.")

    def apply_scaling(self):
        if self.df is None:
            return

        selected_items = self.column_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select one or more columns from the list on the left.")
            return

        selected_columns = [item.text() for item in selected_items]
        method = self.scaling_method_selector.currentText()

        numeric_cols = self.df[selected_columns].select_dtypes(include=np.number).columns.tolist()
        non_numeric_cols = list(set(selected_columns) - set(numeric_cols))

        if not numeric_cols:
            QMessageBox.warning(self, "Warning", "Scaling can only be applied to numeric columns. No numeric columns were selected.")
            return
        
        if non_numeric_cols:
            QMessageBox.information(self, "Info", f"Scaling will only be applied to the following numeric columns:\n\n{', '.join(numeric_cols)}")

        scaler_map = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler
        }
        scaler = scaler_map[method]()
        
        df_modified = self.df.copy()
        plotter = Plotter(self.df)
        changed_cols = []

        for col in numeric_cols:
            transformed_series = pd.Series(
                scaler.fit_transform(self.df[[col]]).flatten(), 
                index=self.df.index,
                name=col
            )
            
            is_accepted = plotter.plot_comparison(self.df[col], transformed_series, col)
            if is_accepted:
                df_modified[col] = transformed_series
                changed_cols.append(col)

        if not changed_cols:
            QMessageBox.information(self, "Info", "No scaling changes were applied (or all were canceled).")
            return
            
        self.df = df_modified
        
        self.operation_history.append({
            'name': 'apply_scaling',
            'columns': changed_cols,
            'method': method
        })

        QMessageBox.information(self, "Success", f"{method} applied to {len(changed_cols)} columns.")
        
        self.update_data_preview()
        self.update_eda_info() 