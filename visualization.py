import seaborn as sns
from mpl_canvas import MplCanvas
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtWidgets import QDialog
from dialogs import ComparisonDialog

# Revert to default style for white background plots
plt.style.use('default')

class Plotter:
    def __init__(self, df):
        self.df = df

    def _get_plot_canvas(self, parent=None):
        return MplCanvas(parent, width=5, height=4, dpi=100)

    def _setup_new_plot(self, canvas):
        """Clears the entire figure and sets up a fresh Axes object for a new plot."""
        canvas.figure.clear()
        canvas.axes = canvas.figure.add_subplot(111)
        
        # White background setup
        canvas.figure.patch.set_facecolor('white')
        canvas.axes.set_facecolor('white')
        
        # Adjust text and spine colors for white background
        canvas.axes.title.set_color('black')
        canvas.axes.xaxis.label.set_color('black')
        canvas.axes.yaxis.label.set_color('black')
        canvas.axes.tick_params(axis='x', colors='black')
        canvas.axes.tick_params(axis='y', colors='black')
        for spine in canvas.axes.spines.values():
            spine.set_edgecolor('black')

    def plot_histogram(self, canvas, column):
        self._setup_new_plot(canvas)
        sns.histplot(self.df[column], ax=canvas.axes, kde=True)
        canvas.axes.set_title(f'Histogram of {column}')
        canvas.figure.tight_layout() # Adjust layout
        canvas.draw()

    def plot_boxplot(self, canvas, column):
        self._setup_new_plot(canvas)
        sns.boxplot(y=self.df[column], ax=canvas.axes)
        canvas.axes.set_title(f'Box Plot of {column}')
        canvas.figure.tight_layout() # Adjust layout
        canvas.draw()

    def plot_barplot(self, canvas, column):
        self._setup_new_plot(canvas)
        if self.df[column].dtype == 'object' or self.df[column].dtype.name == 'category':
            self.df[column].value_counts().plot(kind='bar', ax=canvas.axes)
        else:
             self.df[column].value_counts().plot(kind='bar', ax=canvas.axes)
        canvas.axes.set_title(f'Bar Plot of {column}')
        canvas.figure.tight_layout() # Adjust layout
        canvas.draw()

    def plot_scatter(self, canvas, x_col, y_col, hue_col):
        self._setup_new_plot(canvas)
        
        hue_data = self.df.get(hue_col)
        is_hue_numeric = hue_data is not None and pd.api.types.is_numeric_dtype(hue_data)
        
        if is_hue_numeric:
            # Check if there are any valid (non-NaN) values to use for hue
            if hue_data.notna().sum() == 0:
                sns.scatterplot(x=self.df[x_col], y=self.df[y_col], ax=canvas.axes)
                canvas.axes.text(0.5, 0.5, f'Hue column "{hue_col}"\ncontains only missing values.', 
                                 horizontalalignment='center', verticalalignment='center', 
                                 transform=canvas.axes.transAxes, bbox=dict(facecolor='white', alpha=0.5))
            else:
                palette = "crest"
                sns.scatterplot(x=self.df[x_col], y=self.df[y_col], 
                                hue=hue_data, 
                                ax=canvas.axes,
                                palette=palette,
                                legend=False)
                
                # Create a colorbar manually
                norm = plt.Normalize(hue_data.min(), hue_data.max())
                sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
                sm.set_array([])
                
                # A more robust way to add and style the colorbar
                cbar = canvas.figure.colorbar(sm, ax=canvas.axes)
                cbar.set_label(hue_col, color='black')
                cbar.ax.tick_params(colors='black') # Safer way to set tick color

        else:
            # For categorical data or no hue
            sns.scatterplot(x=self.df[x_col], y=self.df[y_col], 
                               hue=hue_data, 
                               ax=canvas.axes)

        canvas.axes.set_title(f'Scatter Plot: {x_col} vs {y_col}')
        canvas.figure.tight_layout()
        canvas.draw()

    def plot_correlation_heatmap(self, canvas):
        self._setup_new_plot(canvas)
        
        numeric_df = self.df.select_dtypes(include='number')
        corr_matrix = numeric_df.corr()
        
        # Only show annotations and labels if the matrix is not too large
        is_large_matrix = len(corr_matrix.columns) >= 20
        
        sns.heatmap(corr_matrix, annot=not is_large_matrix, fmt=".2f", cmap='coolwarm', ax=canvas.axes)
        canvas.axes.set_title('Correlation Heatmap of Numeric Features')

        # Rotate x-axis labels to give them more space
        plt.setp(canvas.axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Hide y-axis labels if matrix is large
        if is_large_matrix:
            canvas.axes.tick_params(axis='y', labelleft=False)

        canvas.figure.tight_layout()
        canvas.draw()

    def plot_comparison(self, before_series, after_series, column_name):
        """
        Creates and shows a dialog with a side-by-side comparison plot.

        Args:
            before_series (pd.Series): The data before the transformation.
            after_series (pd.Series): The data after the transformation.
            column_name (str): The name of the column being transformed.

        Returns:
            bool: True if the user clicks "OK", False otherwise.
        """
        dialog = ComparisonDialog()
        dialog.setWindowTitle(f"'{column_name}' | Before vs. After")

        # Plot "Before"
        dialog.before_canvas.axes.set_title(f"Before (Original)")
        sns.histplot(before_series, ax=dialog.before_canvas.axes, kde=True)
        dialog.before_canvas.draw()
        
        # Plot "After"
        dialog.after_canvas.axes.set_title(f"After (Transformed)")
        sns.histplot(after_series, ax=dialog.after_canvas.axes, kde=True)
        dialog.after_canvas.draw()
        
        # Show the dialog and return the result
        result = dialog.exec_()
        return result == QDialog.Accepted 