import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import os

'''
    This code loads and visualize the loss over training and validation, and make several graphics from the
    training, validation and testing files. Is used to generate all the necessary graphics used in the report.
'''

os.makedirs('./graphics', exist_ok=True)
models = './models_v2'

class LossVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Loss Visualizer")
        self.root.geometry("900x700")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create Losses tab
        self.losses_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.losses_frame, text="Losses")
        
        # Create Metrics tab
        self.metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_frame, text="Metrics")
        
        # Create Metrics Graphics tab
        self.metrics_graphics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_graphics_frame, text="Metrics Graphics")

        
        # Setup scrollable frames for each tab
        self.setup_losses_tab()
        self.setup_metrics_tab()
        self.setup_metrics_graphics_tab()
 
        # Load and plot the data
        self.load_and_plot_data()
        
    def setup_scrollable_frame(self, parent):
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Bind mousewheel to scroll
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        return scrollable_frame
        
    def setup_losses_tab(self):
        self.losses_scrollable_frame = self.setup_scrollable_frame(self.losses_frame)
        
    def setup_metrics_tab(self):
        self.metrics_scrollable_frame = self.setup_scrollable_frame(self.metrics_frame)
        
    def setup_metrics_graphics_tab(self):
        self.metrics_graphics_scrollable_frame = self.setup_scrollable_frame(self.metrics_graphics_frame)
    

    def create_metrics_table(self, df, metric_name):        
        table_frame = ttk.LabelFrame(self.metrics_scrollable_frame, text=metric_name, padding=10)
        table_frame.pack(fill='x', padx=10, pady=5)
        
        # Create headers
        ttk.Label(table_frame, text="Row Index", width=15).grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(table_frame, text="Training", width=15).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(table_frame, text="Validation", width=15).grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(table_frame, text="Testing", width=15).grid(row=0, column=2, padx=5, pady=2)
        
        # Add data rows
        for idx, row in df.iterrows():
            ttk.Label(table_frame, text=str(idx)).grid(row=idx+1, column=0, padx=5, pady=1)
            ttk.Label(table_frame, text=f"{row['training']:.4f}").grid(row=idx+1, column=1, padx=5, pady=1)
            ttk.Label(table_frame, text=f"{row['validation']:.4f}").grid(row=idx+1, column=2, padx=5, pady=1)
            ttk.Label(table_frame, text=f"{row['testing']:.4f}").grid(row=idx+1, column=2, padx=5, pady=1)

            
    def load_and_plot_data(self):
        # Load training and validation data for losses
        with open(f'{models}/training_loss.json', 'r') as f:
            training_data = json.load(f)
        with open(f'{models}/validation_loss.json', 'r') as f:
            validation_data = json.load(f)
        
        # Load metrics data
        metrics_df = pd.read_csv(f'{models}/training_metrics.csv')
        val_metrics = pd.read_csv(f'{models}/validation_metrics.csv')


        # Load testing data
        tmetrics_df = pd.read_csv(f'{models}/testing_result.csv')
        

        # Plot losses
        loss_types = ['multitask', 'gender', 'hat', 'bag']
        for loss_type in loss_types:
            fig, ax = plt.subplots(figsize=(8, 4))
            epochs = range(1, len(training_data[loss_type]) + 1)
            
            ax.plot(epochs, training_data[loss_type], 'b-', label='Training')
            ax.plot(epochs, validation_data[loss_type], 'r-', label='Validation')
            
            ax.set_title(f'{loss_type.capitalize()} Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            canvas = FigureCanvasTkAgg(fig, master=self.losses_scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10, padx=10)
        
        # Create metrics tables
        metrics = ['Gender', 'Bag', 'Hat']
        metrics_suffixes = ['Accuracy', 'Precision', 'Recall', 'F1', 'Balanced_Accuracy']
        
        for metric in metrics:
            for suffix in metrics_suffixes:
                column_name = f"{metric}_{suffix}"
                df = pd.DataFrame({
                    'training': metrics_df[column_name],
                    'validation': val_metrics[column_name],
                    'testing' : tmetrics_df[column_name]
                })

                self.create_metrics_table(df, column_name)
        
        # Create metrics graphics
        for metric in metrics:
            for suffix in metrics_suffixes:
                column_name = f"{metric}_{suffix}"
                
                fig, ax = plt.subplots(figsize=(8, 4))
                epochs = range(1, len(metrics_df) + 1)
                
                ax.plot(epochs, metrics_df[column_name], 'b-', label='Training')
                ax.plot(epochs, val_metrics[column_name], 'r-', label='Validation')  # Using same data for demonstration
                ax.plot(epochs, tmetrics_df[column_name], 'y-', label='Testing')  # Using same data for demonstration
                
                ax.set_title(f'{column_name}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()

                plt.savefig(f'{column_name}.png', dpi=1200)
                
                canvas = FigureCanvasTkAgg(fig, master=self.metrics_graphics_scrollable_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(pady=10, padx=10)


if __name__ == "__main__":
    root = ttk.Window(themename="cosmo")
    app = LossVisualizerApp(root)
    root.mainloop()