import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Sample purchase history data (e.g., amount spent and frequency of purchases)
purchase_history = np.array([
    [100, 5],
    [80, 7],
    [200, 3],
    [150, 4],
    [120, 6],
    [90, 5]
])

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(purchase_history)
cluster_labels = kmeans.labels_

# Create a GUI
root = tk.Tk()
root.title("Customer Clustering Dashboard")

# Function to update the scatter plot based on selected cluster
def update_plot():
    cluster = cluster_var.get()
    cluster_indices = np.where(cluster_labels == cluster)[0]
    cluster_data = purchase_history[cluster_indices]

    ax.clear()
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], c='blue')
    ax.set_xlabel('Amount Spent')
    ax.set_ylabel('Frequency of Purchases')
    ax.set_title(f'Cluster {cluster} - Customer Segmentation')
    canvas.draw()

# Creating the scatter plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(purchase_history[:, 0], purchase_history[:, 1], c=cluster_labels, cmap='viridis')
ax.set_xlabel('Amount Spent')
ax.set_ylabel('Frequency of Purchases')
ax.set_title('Customer Segmentation')

# Embedding the plot in tkinter GUI
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0, padx=10, pady=10, rowspan=3)

# Dropdown menu for selecting clusters
cluster_var = tk.IntVar()
cluster_var.set(0)
cluster_dropdown = ttk.Combobox(root, textvariable=cluster_var, values=[0, 1, 2])
cluster_dropdown.grid(row=0, column=1, padx=10, pady=10)

# Button to update plot
update_button = ttk.Button(root, text="Update Plot", command=update_plot)
update_button.grid(row=1, column=1, padx=10, pady=10)

# Label and entry widgets for entering new customer data
amount_label = ttk.Label(root, text="Amount Spent:")
amount_label.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
amount_entry = ttk.Entry(root)
amount_entry.grid(row=2, column=1, padx=10, pady=5, sticky=tk.E)

frequency_label = ttk.Label(root, text="Frequency of Purchases:")
frequency_label.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
frequency_entry = ttk.Entry(root)
frequency_entry.grid(row=3, column=1, padx=10, pady=5, sticky=tk.E)

# Function to add new data point and update the plot
def add_data_point():
    amount = float(amount_entry.get())
    frequency = float(frequency_entry.get())
    new_point = np.array([[amount, frequency]])
    global purchase_history
    purchase_history = np.vstack([purchase_history, new_point])
    global cluster_labels
    kmeans.fit(purchase_history)
    cluster_labels = kmeans.labels_
    update_plot()

# Button to add new data point
add_button = ttk.Button(root, text="Add Data Point", command=add_data_point)
add_button.grid(row=4, column=1, padx=10, pady=10)

# Run the GUI
root.mainloop()