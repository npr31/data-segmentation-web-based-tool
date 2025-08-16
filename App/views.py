import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from django.shortcuts import render,redirect
from io import BytesIO
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.preprocessing import LabelEncoder
from io import StringIO
from django.contrib.auth.models import User
from .models import *
from .models import sales
from django.contrib.auth.decorators import login_required
# from django.contrib.auth import login,logout
from django.contrib.auth import authenticate,login,logout
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
import numpy as np
from collections import Counter
from io import TextIOWrapper
import csv
import json
from django.contrib import messages
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from django.http import JsonResponse
from .ai_analysis import ClusterAnalyzer
from scipy.interpolate import griddata
import plotly
from datetime import datetime
import tempfile
import os
from django.http import HttpResponse

# Set matplotlib to use Agg backend
import matplotlib
matplotlib.use('Agg')

def upload_dataset(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        try:
            profile = Profile.objects.filter(user=request.user).first()
            csv_file = request.FILES['csv_file']
            file_data = csv_file.read().decode('utf-8')
            data = pd.read_csv(StringIO(file_data))

            # Store dataset in session
            if profile and hasattr(profile, 'branch'):
                Dataset.objects.create(branch=profile.branch, dataset=request.FILES['csv_file'])
            else:
                Dataset.objects.create(dataset=request.FILES['csv_file'])
            request.session['dataset'] = data.to_dict()

            return render(request, 'display_dataset.html', {'data': data.to_html()})
        except Exception as e:
            messages.error(request, f"Error uploading file: {str(e)}")
            return redirect('home')
    try:
        profile = Profile.objects.filter(user=request.user).first()
        datasets = Dataset.objects.filter(branch=profile.branch) if profile and hasattr(profile, 'branch') else Dataset.objects.all()
    except:
        profile = ''
        datasets = ''
    return render(request, 'home.html', {"datasets": datasets, "profile": profile})


def upload_dataset2(request):
    if request.method == 'POST':
        profile = Profile.objects.filter(user = request.user).first()
        dataset = Dataset.objects.filter(pk = request.POST.get('csv_file')).first()
        csv_file = dataset.dataset
        file_data = csv_file.read().decode('utf-8')
        data = pd.read_csv(StringIO(file_data))

        # Store dataset in session
        # Dataset.objects.create(branch = profile.branch,dataset = request.FILES['csv_file'])
        request.session['dataset'] = data.to_dict()

        return render(request, 'display_dataset.html', {'data': data.to_html()})
    profile = Profile.objects.filter(user = request.user).first()
    datasets = Dataset.objects.filter(branch = profile.branch)
    return render(request, 'home.html',{"datasets":datasets,"profile":profile})

def sign_up(request):
     error_message=None
     user=None
     if request.POST:
        firstname=request.POST['firstname']
        username=request.POST['username']
        password=request.POST['password']
        email=request.POST['email']
        try:
            user=User.objects.create_user(username=username,password=password,email=email)
            user.first_name = firstname
            user.save()
            login(request, user)  
            return redirect('home')
        except Exception as e: 
            error_message=str(e)   
     return render(request,'signup.html',{'user':user,'error_message':error_message})

 
def login_view(request):
    if request.POST:
        username=request.POST['username']
        password=request.POST['password']  
        user=authenticate(username=username,password=password) 
        if user:
            login(request,user)
            return redirect('home')
        else:           
            return redirect('signup')
    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    return redirect('home')


# def generate_chart(request):
#     if request.method == 'POST':
#         dataset = pd.DataFrame(request.session.get('dataset'))
#         label_encoders = request.session.get('label_encoders', {})

#         x_col = request.POST['x_column']
#         y_col = request.POST['y_column']
#         chart_type = request.POST['chart_type']
#         color = request.POST['color']

       
#         for column in [x_col, y_col]:
#             if column in label_encoders:
#                 encoder = LabelEncoder()
#                 encoder.classes_ = np.array(label_encoders[column])
#                 dataset[column] = encoder.inverse_transform(dataset[column].astype(int))

       
#         data_size = len(dataset)
#         width = min(10, max(5, data_size / 10))  
#         height = min(8, max(5, data_size / 20)) 
        
       
#         sns.set_theme(style="whitegrid")
        
#         fig, ax = plt.subplots(figsize=(width, height))

       
#         if chart_type == 'line':
#             sns.lineplot(x=x_col, y=y_col, data=dataset, ax=ax, palette='deep')
#             ax.set_title(f'Line Chart: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

#         elif chart_type == 'bar':
#             if pd.api.types.is_numeric_dtype(dataset[y_col]):
#                 sns.barplot(x=x_col, y=y_col, data=dataset, ax=ax, palette='coolwarm')
#             else:
#                 sns.countplot(x=x_col, data=dataset, ax=ax, palette='coolwarm')
#             ax.set_title(f'Bar Chart: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

#         elif chart_type == 'scatter':
#             sns.scatterplot(x=x_col, y=y_col, data=dataset, ax=ax, palette='Set2')
#             ax.set_title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

#         elif chart_type == 'histogram':
#             if pd.api.types.is_numeric_dtype(dataset[x_col]):
#                 ax.hist(dataset[x_col], bins=20, color=color, edgecolor='black')
#                 ax.set_title(f'Histogram of {x_col}', fontsize=16, fontweight='bold')
#             else:
#                 return render(request, 'error.html', {'message': 'Histogram requires numeric data for the X column.'})

#         elif chart_type == 'box':
#             sns.boxplot(x=x_col, y=y_col, data=dataset, ax=ax, palette='Set3')
#             ax.set_title(f'Box Plot: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

#         elif chart_type == 'pie':
#             if pd.api.types.is_numeric_dtype(dataset[y_col]):
#                 data_to_plot = dataset[y_col].value_counts()
#                 labels = data_to_plot.index
#                 sizes = data_to_plot.values
                
#                 num_colors = len(labels)
#                 colors = sns.color_palette("pastel", num_colors)

#                 wedges, texts, autotexts = ax.pie(
#                     sizes,
#                     labels=labels,
#                     colors=colors,
#                     autopct='%1.1f%%',
#                     startangle=90,
#                     wedgeprops={'edgecolor': color}
#                 )

#                 for text in texts + autotexts:
#                     text.set_fontsize(12)
#                     text.set_color('black')
#                 ax.set_title(f'Pie Chart of {y_col}', fontsize=16, fontweight='bold')

#         elif chart_type == 'heatmap':
#             if pd.api.types.is_numeric_dtype(dataset[x_col]) and pd.api.types.is_numeric_dtype(dataset[y_col]):
#                 pivot_table = dataset.pivot_table(index=x_col, columns=y_col, aggfunc='size', fill_value=0)
#                 sns.heatmap(pivot_table, ax=ax, cmap='YlGnBu')
#                 ax.set_title(f'Heatmap of {x_col} vs {y_col}', fontsize=16, fontweight='bold')

#         # Set axis labels and grid
#         ax.set_xlabel(x_col, fontsize=12)
#         ax.set_ylabel(y_col, fontsize=12)
#         ax.grid(True, which='both', linestyle='--', linewidth=0.7)

#         # Calculate statistics only for numeric columns
#         x_stats = {}
#         y_stats = {}

#         if pd.api.types.is_numeric_dtype(dataset[x_col]):
#             x_stats = {
#                 'mean': dataset[x_col].mean(),
#                 'median': dataset[x_col].median(),
#                 'mode': dataset[x_col].mode()[0] if not dataset[x_col].mode().empty else 'N/A'
#             }
#         else:
#             x_stats = {'mean': 'N/A', 'median': 'N/A', 'mode': 'N/A'}

#         if pd.api.types.is_numeric_dtype(dataset[y_col]):
#             y_stats = {
#                 'mean': dataset[y_col].mean(),
#                 'median': dataset[y_col].median(),
#                 'mode': dataset[y_col].mode()[0] if not dataset[y_col].mode().empty else 'N/A'
#             }
#         else:
#             y_stats = {'mean': 'N/A', 'median': 'N/A', 'mode': 'N/A'}

#         # Apply tight layout to prevent overlap
#         plt.tight_layout()

#         # Save the figure to a BytesIO buffer and encode it to base64
#         buffer = BytesIO()
#         canvas = FigureCanvasAgg(fig)
#         canvas.draw()
#         buffer.seek(0)
#         fig.savefig(buffer, format='png')
#         buffer.seek(0)
#         image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
#         buffer.close()

#         return render(request, 'chart.html', {
#             'chart': image_base64,
#             'x_stats': x_stats,
#             'y_stats': y_stats
#         })

#     dataset = pd.DataFrame(request.session.get('dataset'))
#     columns = dataset.columns
#     return render(request, 'select_columns.html', {'columns': columns})



def generate_chart(request):
    if request.method == 'POST':
        dataset = pd.DataFrame(request.session.get('dataset'))
        label_encoders = request.session.get('label_encoders', {})

        x_col = request.POST['x_column']
        y_col = request.POST['y_column']
        chart_type = request.POST['chart_type']
        color = request.POST.get('color', '#1f77b4')
        z_col = request.POST.get('z_column')
        size_col = request.POST.get('size_column')

        # Decoding labels if needed
        for column in [x_col, y_col, z_col, size_col]:
            if column and column in label_encoders:
                encoder = LabelEncoder()
                encoder.classes_ = np.array(label_encoders[column])
                dataset[column] = encoder.inverse_transform(dataset[column].astype(int))

        try:
            if chart_type in ['3d_scatter', 'surface']:
                if not z_col:
                    return render(request, 'error.html', 
                                {'message': '3D visualization requires a Z-axis column.'})
                
                # Check if all columns are numeric
                if not all(pd.api.types.is_numeric_dtype(dataset[col]) for col in [x_col, y_col, z_col]):
                    return render(request, 'error.html', 
                                {'message': '3D visualization requires numeric data for all axes.'})

                if chart_type == '3d_scatter':
                    # Create 3D scatter plot using Plotly
                    fig = {
                        'data': [{
                            'type': 'scatter3d',
                            'mode': 'markers',
                            'x': dataset[x_col].tolist(),
                            'y': dataset[y_col].tolist(),
                            'z': dataset[z_col].tolist(),
                            'marker': {
                                'size': 5,
                                'color': dataset[z_col].tolist(),
                                'colorscale': 'Viridis',
                                'showscale': True
                            }
                        }],
                        'layout': {
                            'title': f'3D Scatter Plot: {x_col} vs {y_col} vs {z_col}',
                            'scene': {
                                'xaxis': {'title': x_col},
                                'yaxis': {'title': y_col},
                                'zaxis': {'title': z_col}
                            },
                            'margin': {'l': 0, 'r': 0, 'b': 0, 't': 40}
                        }
                    }
                else:  # surface plot
                    # Create a grid of points
                    x = np.linspace(dataset[x_col].min(), dataset[x_col].max(), 100)
                    y = np.linspace(dataset[y_col].min(), dataset[y_col].max(), 100)
                    X, Y = np.meshgrid(x, y)
                    
                    # Interpolate Z values
                    points = np.column_stack((dataset[x_col], dataset[y_col]))
                    Z = griddata(points, dataset[z_col], (X, Y), method='cubic', fill_value=0)
                    
                    # Check if interpolation was successful
                    if np.isnan(Z).any():
                        return render(request, 'error.html', 
                                    {'message': 'Could not create surface plot. Data points may be too sparse or irregular.'})
                    
                    # Create surface plot using Plotly
                    fig = {
                        'data': [{
                            'type': 'surface',
                            'x': X.tolist(),
                            'y': Y.tolist(),
                            'z': Z.tolist(),
                            'colorscale': 'Viridis'
                        }],
                        'layout': {
                            'title': f'3D Surface Plot: {x_col} vs {y_col} vs {z_col}',
                            'scene': {
                                'xaxis': {'title': x_col},
                                'yaxis': {'title': y_col},
                                'zaxis': {'title': z_col}
                            },
                            'margin': {'l': 0, 'r': 0, 'b': 0, 't': 40}
                        }
                    }

                # Convert the figure to JSON
                chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return render(request, 'chart.html', {
                    'chart_json': chart_json,
                    'is_3d': True
                })

            # Calculate size for the chart based on data length
            data_size = len(dataset)
            width = min(10, max(5, data_size / 10))  
            height = min(8, max(5, data_size / 20)) 
            
            # Set the seaborn style
            sns.set_theme(style="whitegrid")
            
            fig, ax = plt.subplots(figsize=(width, height))

            try:
                if chart_type == 'line':
                    sns.lineplot(x=x_col, y=y_col, data=dataset, ax=ax, color=color)
                    ax.set_title(f'Line Chart: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

                elif chart_type == 'bar':
                    if pd.api.types.is_numeric_dtype(dataset[y_col]):
                        sns.barplot(x=x_col, y=y_col, data=dataset, ax=ax, color=color)
                    else:
                        sns.countplot(x=x_col, data=dataset, ax=ax, color=color)
                    ax.set_title(f'Bar Chart: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

                elif chart_type == 'scatter':
                    sns.scatterplot(x=x_col, y=y_col, data=dataset, ax=ax, color=color)
                    ax.set_title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

                elif chart_type == 'histogram':
                    if pd.api.types.is_numeric_dtype(dataset[x_col]):
                        ax.hist(dataset[x_col], bins=20, color=color, edgecolor='black')
                        ax.set_title(f'Histogram of {x_col}', fontsize=16, fontweight='bold')
                    else:
                        return render(request, 'error.html', {'message': 'Histogram requires numeric data for the X column.'})

                elif chart_type == 'box':
                    sns.boxplot(x=x_col, y=y_col, data=dataset, ax=ax, color=color)
                    ax.set_title(f'Box Plot: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

                elif chart_type == 'pie':
                    if pd.api.types.is_numeric_dtype(dataset[y_col]):
                        data_to_plot = dataset[y_col].value_counts()
                        labels = data_to_plot.index
                        sizes = data_to_plot.values
                        
                        num_colors = len(labels)
                        colors = sns.color_palette("pastel", num_colors)

                        wedges, texts, autotexts = ax.pie(
                            sizes,
                            labels=labels,
                            colors=colors,
                            autopct='%1.1f%%',
                            startangle=90,
                            wedgeprops={'edgecolor': color}
                        )

                        for text in texts + autotexts:
                            text.set_fontsize(12)
                            text.set_color('black')
                        ax.set_title(f'Pie Chart of {y_col}', fontsize=16, fontweight='bold')

                elif chart_type == 'heatmap':
                    # Create a pivot table for the heatmap
                    pivot_data = dataset.pivot_table(
                        values=y_col,
                        index=x_col,
                        aggfunc='mean'
                    )
                    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', ax=ax)
                    ax.set_title(f'Heatmap: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

                elif chart_type == 'violin':
                    sns.violinplot(x=x_col, y=y_col, data=dataset, ax=ax, color=color)
                    ax.set_title(f'Violin Plot: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

                elif chart_type == 'area':
                    dataset.plot(x=x_col, y=y_col, kind='area', ax=ax, color=color)
                    ax.set_title(f'Area Chart: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

                elif chart_type == 'bubble':
                    if size_col:
                        plt.scatter(dataset[x_col], dataset[y_col], 
                                  s=dataset[size_col]*100, alpha=0.5, color=color)
                        ax.set_title(f'Bubble Chart: {x_col} vs {y_col} (size: {size_col})', 
                                   fontsize=16, fontweight='bold')
                    else:
                        return render(request, 'error.html', 
                                    {'message': 'Bubble chart requires a size column.'})

                elif chart_type == 'radar':
                    if len(dataset[x_col].unique()) > 10:
                        return render(request, 'error.html', 
                                    {'message': 'Radar chart requires fewer than 10 categories.'})
                    
                    categories = dataset[x_col].unique()
                    values = dataset.groupby(x_col)[y_col].mean()
                    
                    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
                    values = np.concatenate((values, [values[0]]))
                    angles = np.concatenate((angles, [angles[0]]))
                    
                    ax.plot(angles, values, color=color)
                    ax.fill(angles, values, color=color, alpha=0.25)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)
                    ax.set_title(f'Radar Chart: {x_col} vs {y_col}', fontsize=16, fontweight='bold')

                # Set axis labels and grid
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel(y_col, fontsize=12)
                plt.xticks(rotation=90)
                ax.grid(True, which='both', linestyle='--', linewidth=0.7)

                # Calculate statistics for numeric columns
                x_stats = {}
                y_stats = {}

                if pd.api.types.is_numeric_dtype(dataset[x_col]):
                    x_stats = {
                        'mean': dataset[x_col].mean(),
                        'median': dataset[x_col].median(),
                        'mode': dataset[x_col].mode()[0] if not dataset[x_col].mode().empty else 'N/A'
                    }
                else:
                    x_stats = {'mean': 'N/A', 'median': 'N/A', 'mode': 'N/A'}

                if pd.api.types.is_numeric_dtype(dataset[y_col]):
                    y_stats = {
                        'mean': dataset[y_col].mean(),
                        'median': dataset[y_col].median(),
                        'mode': dataset[y_col].mode()[0] if not dataset[y_col].mode().empty else 'N/A'
                    }
                else:
                    y_stats = {'mean': 'N/A', 'median': 'N/A', 'mode': 'N/A'}

                # Apply tight layout to prevent overlap
                plt.tight_layout()

                # Save the figure to a BytesIO buffer and encode it to base64
                buffer = BytesIO()
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                buffer.seek(0)
                fig.savefig(buffer, format='png')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                buffer.close()

                return render(request, 'chart.html', {
                    'chart': image_base64,
                    'x_stats': x_stats,
                    'y_stats': y_stats
                })

            except Exception as e:
                return render(request, 'error.html', {'message': f'Error generating chart: {str(e)}'})

        except Exception as e:
            return render(request, 'error.html', {'message': f'Error generating chart: {str(e)}'})

    dataset = pd.DataFrame(request.session.get('dataset'))
    columns = dataset.columns
    return render(request, 'select_columns.html', {'columns': columns})

# sales page

def Sales(request):
     error_message=None
     user=None
     if request.POST:
        firstname=request.POST['firstname']
        username=request.POST['username']
        password=request.POST['password']
        phonenumber=request.POST['phonenumber']
        email=request.POST['email']
        address=request.POST['address']
        branchname=request.POST['branchname']
        try:
            user=User.objects.create_user(username=username,password=password,email=email)
            user.first_name = firstname
            user.save()
            Profile.objects.create(user=user, phone_number=phonenumber,address = address,branch = branchname) 
            sales.objects.create(user=user, phone_number=phonenumber,branch=branchname,address=address)
            login(request, user)
            return redirect('home')
        except Exception as e: 
            error_message=str(e) 
     profiles = Profile.objects.filter(is_branchmanager = True)
     return render(request,'sales.html',{'user':user,'error_message':error_message,"profiles":profiles})
from django.shortcuts import render
from collections import Counter
from io import TextIOWrapper
import csv
import json

def dashboard_view(request):
    sales_data = []
    total_sales = 0
    most_demanded_product = "N/A"
    total_customers = 0
    total_products_sold = 0
    top_paying_customer = "N/A"
    highest_sales_branch = "N/A"
    product_sales = Counter()

    if request.method == "POST" and request.FILES.get("csv_file"):
        csv_file = request.FILES["csv_file"]
        csv_data = TextIOWrapper(csv_file.file, encoding="utf-8")
        reader = csv.DictReader(csv_data)

        sales_data = list(reader)

        # Calculate metrics
        total_sales = sum(float(row["Final Amount"]) for row in sales_data)
        customer_spend = Counter()
        branch_sales = Counter()

        for row in sales_data:
            product = row["Product Name"]
            product_sales[product] += int(row["Quantity"])
            customer_spend[row["Customer Name"]] += float(row["Final Amount"])
            branch_sales[row["Branch Name"]] += float(row["Final Amount"])

        # Calculate metrics
        most_demanded_product = product_sales.most_common(1)[0][0] if product_sales else "N/A"
        total_customers = len(customer_spend)
        total_products_sold = sum(product_sales.values())
        top_paying_customer = customer_spend.most_common(1)[0][0] if customer_spend else "N/A"
        highest_sales_branch = branch_sales.most_common(1)[0][0] if branch_sales else "N/A"

    # Prepare data for the chart
    product_names = list(product_sales.keys())
    product_quantities = list(product_sales.values())
    profile = Profile.objects.filter(user = request.user).first()
    datasets = Dataset.objects.filter(branch = profile.branch)
    context = {
        "total_sales": total_sales,
        "most_demanded_product": most_demanded_product,
        "total_customers": total_customers,
        "total_products_sold": total_products_sold,
        "top_paying_customer": top_paying_customer,
        "highest_sales_branch": highest_sales_branch,
        "product_names": json.dumps(product_names),  # Serialize for frontend
        "product_quantities": json.dumps(product_quantities), 
         "datasets":datasets # Serialize for frontend
    }
  
    return render(request, "dashboard.html", context)

@login_required(login_url='login')
def prediction_view(request):
    try:
        profile = Profile.objects.filter(user=request.user).first()
        datasets = Dataset.objects.all()  # Initialize datasets with all datasets
        
        if profile and hasattr(profile, 'branch'):
            datasets = Dataset.objects.filter(branch=profile.branch)
        
        if request.method == 'POST' and request.FILES['csv_file']:
            csv_file = request.FILES['csv_file']
            file_data = csv_file.read().decode('utf-8')
            data = pd.read_csv(StringIO(file_data))

            # Drop duration column as it's not available during prediction
            if 'duration' in data.columns:
                data = data.drop('duration', axis=1)
            
            # Cap age at 70
            data['age'] = data['age'].apply(lambda x: 70 if x > 70 else x)
            
            # Convert response to binary
            if 'response' in data.columns:
                data['response'] = data['response'].map({'yes': 1, 'no': 0})
            
            # Feature engineering
            # Education categories
            data.replace({'education': {
                'basic.9y': 'Primary_Education',
                'basic.4y': 'Primary_Education',
                'basic.6y': 'Primary_Education',
                'illiterate': 'Primary_Education',
                'high.school': 'Secondary_Education',
                'university.degree': 'Tertiary_Education'
            }}, inplace=True)
            
            # Drop default column if exists
            if 'default' in data.columns:
                data = data.drop('default', axis=1)
            
            # Process pdays
            if 'pdays' in data.columns:
                data['pdays'] = data['pdays'].apply(lambda x: 'contacted_in_first_10_days' if x in range(11) 
                                                  else ('contacted_first_time' if x == 999 
                                                       else 'contacted_after_10_days'))
            
            # Process previous contacts
            if 'previous' in data.columns:
                data['previous'] = data['previous'].apply(lambda x: 'never_contacted' if x == 0 
                                                        else ('less_than_3_times' if x <= 3 
                                                             else 'more_than_3_times'))
            
            # Create dummy variables
            categorical_cols = ['job', 'marital', 'education', 'housing', 'loan', 
                              'contact', 'month', 'day_of_week', 'poutcome', 'previous', 'pdays']
            data_encoded = pd.get_dummies(data, columns=[col for col in categorical_cols if col in data.columns])
            
            # Scale numeric features
            numeric_cols = ['age', 'campaign', 'emp.var.rate', 'cons.price.idx', 
                          'cons.conf.idx', 'euribor3m', 'nr.employed']
            numeric_cols = [col for col in numeric_cols if col in data_encoded.columns]
            
            if numeric_cols:
                scaler = StandardScaler()
                data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])
            
            # PCA transformation
            pca = PCA(n_components=15)
            X_pca = pca.fit_transform(data_encoded.drop('response', axis=1) if 'response' in data_encoded.columns else data_encoded)
            
            # Train logistic regression
            if 'response' in data_encoded.columns:
                X = X_pca
                y = data_encoded['response']
                
                model = LogisticRegression(class_weight='balanced')
                model.fit(X, y)
                
                # Get predictions and probabilities
                y_pred = model.predict(X)
                y_pred_prob = model.predict_proba(X)[:, 1]
                
                # Calculate metrics
                cm = confusion_matrix(y, y_pred)
                sensitivity = recall_score(y, y_pred)  # Same as recall for binary classification
                specificity = recall_score(y, y_pred, pos_label=0)  # Recall of the negative class
                auc_score = roc_auc_score(y, y_pred_prob)
                
                # Create deciles for gain and lift charts
                df_proba = pd.DataFrame({
                    'actual': y,
                    'pred_proba': y_pred_prob,
                    'predicted': y_pred
                })
                
                df_proba['decile'] = pd.qcut(df_proba['pred_proba'], 10, np.arange(10, 0, -1))
                
                # Calculate lift and gain
                df_lift = df_proba.groupby('decile')['pred_proba'].count().reset_index()
                df_lift_pred = df_proba[df_proba['actual']==1].groupby('decile')['actual'].count().reset_index()
                df_lift_final = df_lift.merge(df_lift_pred, on='decile')
                df_lift_final = df_lift_final.sort_values('decile', ascending=False)
                
                df_lift_final['cum_response'] = df_lift_final['actual'].cumsum()
                df_lift_final['gain'] = (100 * df_lift_final['cum_response'] / df_lift_final['actual'].sum()).round(2)
                df_lift_final['cum_lift'] = (df_lift_final['gain'] / (df_lift_final['decile'].astype('int') * 10)).round(2)
                
                # Create visualizations
                # Gain Chart
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, 11), df_lift_final['gain'], marker='o')
                plt.title('Gain Chart')
                plt.xlabel('Decile')
                plt.ylabel('Cumulative Gain (%)')
                plt.grid(True)
                gain_buffer = BytesIO()
                plt.savefig(gain_buffer, format='png', bbox_inches='tight')
                gain_buffer.seek(0)
                gain_chart = base64.b64encode(gain_buffer.getvalue()).decode('utf-8')
                plt.close()
                
                # Lift Chart
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, 11), df_lift_final['cum_lift'], marker='o')
                plt.title('Lift Chart')
                plt.xlabel('Decile')
                plt.ylabel('Cumulative Lift')
                plt.grid(True)
                lift_buffer = BytesIO()
                plt.savefig(lift_buffer, format='png', bbox_inches='tight')
                lift_buffer.seek(0)
                lift_chart = base64.b64encode(lift_buffer.getvalue()).decode('utf-8')
                plt.close()

                # Identify high-value customers
                high_value_customers = data[
                    (y_pred_prob > np.mean(y_pred_prob)) &  # Above average predicted probability
                    (y_pred_prob > 0.7)  # High probability of conversion
                ].copy()  # Create a copy to avoid SettingWithCopyWarning

                # Add prediction probabilities and actual predictions to the export
                high_value_customers['predicted_probability'] = y_pred_prob[
                    (y_pred_prob > np.mean(y_pred_prob)) & 
                    (y_pred_prob > 0.7)
                ]
                high_value_customers['predicted_response'] = y_pred[
                    (y_pred_prob > np.mean(y_pred_prob)) & 
                    (y_pred_prob > 0.7)
                ]

                # Define the exact order of columns we want in the output
                output_columns = [
                    'age', 'job', 'marital', 'education', 'housing', 'loan',
                    'contact', 'month', 'day_of_week', 'campaign', 'pdays',
                    'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
                    'cons.conf.idx', 'euribor3m', 'nr.employed', 'response',
                    'predicted_probability', 'predicted_response'
                ]

                # Ensure all required columns are present and in the correct order
                for col in output_columns:
                    if col not in high_value_customers.columns:
                        if col in ['predicted_probability', 'predicted_response']:
                            continue  # Skip if these columns are already handled
                        elif col in ['age', 'campaign', 'emp.var.rate', 'cons.price.idx', 
                                   'cons.conf.idx', 'euribor3m', 'nr.employed']:
                            high_value_customers[col] = data[col].mean()  # Use mean for numeric columns
                        else:
                            high_value_customers[col] = data[col].mode()[0]  # Use mode for categorical columns

                # Reorder columns to match the expected order
                high_value_customers = high_value_customers[output_columns]

                # Generate a unique filename for the CSV
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f'high_value_customers_{timestamp}.csv'
                
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                high_value_customers.to_csv(temp_file.name, index=False)
                
                # Store the file path in session for later download
                request.session['high_value_customers_file'] = temp_file.name
                
                # Prepare stats for display
                stats = {
                    'total_customers': len(data),
                    'response_rate': float(np.round(y.mean() * 100, 2)),
                    'model_accuracy': float(np.round(model.score(X, y) * 100, 2)),
                    'sensitivity': float(np.round(sensitivity * 100, 2)),
                    'specificity': float(np.round(specificity * 100, 2)),
                    'auc_score': float(np.round(auc_score, 2)),
                    'gain_at_50': float(df_lift_final.iloc[4]['gain']),
                    'lift_at_50': float(df_lift_final.iloc[4]['cum_lift'])
                }
                
                return render(request, 'prediction.html', {
                    'datasets': datasets,
                    'profile': profile,
                    'stats': stats,
                    'gain_chart': gain_chart,
                    'lift_chart': lift_chart,
                    'analysis_complete': True,
                    'download_available': True,
                    'high_value_count': len(high_value_customers),
                    'total_customers': len(data),
                    'output_filename': output_filename
                })
                
    except Exception as e:
        print(f"Error: {str(e)}")
        messages.error(request, f"An error occurred: {str(e)}")
        
    return render(request, 'prediction.html', {
        'datasets': datasets,
        'profile': profile
    })

def download_high_value_customers(request):
    try:
        # Get the file path from session
        file_path = request.session.get('high_value_customers_file')
        if not file_path or not os.path.exists(file_path):
            return HttpResponse('File not found', status=404)

        # Read the file and create response
        with open(file_path, 'rb') as f:
            file_content = f.read()
            
        # Create the response
        response = HttpResponse(file_content, content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename=high_value_customers.csv'
        
        # Clean up the temporary file after reading
        try:
            os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file: {str(e)}")
        
        # Remove the file path from session
        if 'high_value_customers_file' in request.session:
            del request.session['high_value_customers_file']
        
        return response
            
    except Exception as e:
        return HttpResponse(str(e), status=500)

@login_required(login_url='login')
def segmentation_view(request):
    if request.method == 'POST':
        # If user is dropping nulls (no file upload, just drop_nulls in POST)
        if request.POST.get('drop_nulls') and 'csv_file' not in request.FILES:
            # Load the DataFrame from session
            dataset = request.session.get('dataset')
            if dataset:
                df = pd.DataFrame(dataset)
                original_rows = len(df)
                df = df.dropna()
                remaining_rows = len(df)
                
                # If more than 50% of rows are dropped, show a warning
                if remaining_rows < original_rows * 0.5:
                    messages.warning(request, f"Warning: Dropping null values removed {original_rows - remaining_rows} rows ({round((original_rows - remaining_rows)/original_rows*100, 2)}% of data). Proceeding with remaining {remaining_rows} rows.")
                
                # If no rows remain, show error
                if remaining_rows == 0:
                    messages.error(request, "Dropping null values would result in an empty dataset. Please handle null values differently or upload a new dataset.")
                    return render(request, 'segmentation.html')
                
                request.session['dataset'] = df.to_dict()
                return redirect('segmentation_result')
            else:
                messages.error(request, "No dataset found in session. Please upload a file first.")
                return render(request, 'segmentation.html')
        # Normal file upload flow
        if 'csv_file' not in request.FILES:
            messages.error(request, "Please select a CSV file to upload.")
            return render(request, 'segmentation.html')
        csv_file = request.FILES['csv_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, "Please upload a valid CSV file.")
            return render(request, 'segmentation.html')
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            has_nulls = df.isnull().values.any()
            request.session['dataset'] = df.to_dict()
            if has_nulls and not request.POST.get('drop_nulls'):
                # Show message and button to drop nulls
                return render(request, 'segmentation.html', {
                    'nulls_detected': True,
                    'null_count': int(df.isnull().sum().sum()),
                    'columns_with_nulls': df.isnull().sum().to_dict(),
                })
            # If user chose to drop nulls, do it
            if request.POST.get('drop_nulls'):
                df = df.dropna()
                request.session['dataset'] = df.to_dict()
            # Redirect to segmentation result view
            return redirect('segmentation_result')
        except Exception as e:
            messages.error(request, f"Error processing file: {str(e)}")
            return render(request, 'segmentation.html')
    return render(request, 'segmentation.html')

def help_view(request):
    try:
        profile = Profile.objects.filter(user=request.user).first() if request.user.is_authenticated else None
    except:
        profile = None
    return render(request, 'help.html', {
        "profile": profile
    })

def contact_view(request):
    try:
        profile = Profile.objects.filter(user=request.user).first() if request.user.is_authenticated else None
    except:
        profile = None
    return render(request, 'contact.html', {
        "profile": profile
    })

@login_required(login_url='login')
def segmentation_result_view(request):
    try:
        # Get the dataset from the session
        dataset = request.session.get('dataset')
        if not dataset:
            messages.error(request, "No dataset found in session. Please upload a file first.")
            return redirect('segmentation')
        
        # Convert back to DataFrame
        data = pd.DataFrame(dataset)
        
        # Check if dataset is empty
        if data.empty:
            messages.error(request, "The dataset is empty. Please upload a new dataset.")
            return redirect('segmentation')
            
        # Prepare data preview
        head_data = data.head().to_html(classes='table table-striped table-bordered', index=True)
        tail_data = data.tail().to_html(classes='table table-striped table-bordered', index=True)
        
        # Prepare dataset information
        total_rows = len(data)
        total_features = len(data.columns)
        
        # Detailed cleaning information
        null_info = {
            'total_nulls': data.isnull().sum().sum(),
            'columns_with_nulls': {
                col: count for col, count in data.isnull().sum().items() if count > 0
            },
            'has_nulls': data.isnull().sum().sum() > 0
        }
        
        duplicate_info = {
            'total_duplicates': data.duplicated().sum(),
            'has_duplicates': data.duplicated().sum() > 0
        }

        # Calculate dataset description statistics
        description_stats = data.describe().round(2)
        description_html = description_stats.to_html(classes='table table-striped table-bordered', index=True)
        
        # Check if we have numerical data
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) == 0:
            messages.error(request, "No numerical columns found in the dataset. Please ensure your CSV contains numerical data.")
            return redirect('segmentation')
        
        # Get number of clusters from request or default to 3
        n_clusters = int(request.POST.get('n_clusters', 3))
        color_scale = request.POST.get('color_scale', 'viridis')
        
        # Prepare data for clustering
        X = data[numerical_cols]
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for 2D and 3D visualization
        pca_2d = PCA(n_components=2)
        pca_3d = PCA(n_components=3)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        X_pca_3d = pca_3d.fit_transform(X_scaled)
        
        # Calculate WCSS and Silhouette Score for different numbers of clusters
        k_clusters = 8  # Maximum number of clusters to try
        wcss = []
        silhouette_scores = []
        
        for i in range(2, k_clusters + 1):
            kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            
            # Calculate silhouette score
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(X_pca_2d, labels)
            silhouette_scores.append(round(silhouette_avg, 5))
        
        # Find optimal number of clusters
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Use optimal_k as the default number of clusters unless overridden by POST
        n_clusters = int(request.POST.get('n_clusters', optimal_k))
        color_scale = request.POST.get('color_scale', 'viridis')

        # Create WCSS plot (Elbow method)
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, k_clusters + 1), wcss, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
        plt.grid(True)
        wcss_buffer = BytesIO()
        plt.savefig(wcss_buffer, format='png', bbox_inches='tight')
        wcss_buffer.seek(0)
        wcss_plot = base64.b64encode(wcss_buffer.getvalue()).decode('utf-8')
        plt.close()

        # Create Silhouette Score plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, k_clusters + 1), silhouette_scores, marker='o')
        plt.title('Silhouette Score Analysis')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        silhouette_buffer = BytesIO()
        plt.savefig(silhouette_buffer, format='png', bbox_inches='tight')
        silhouette_buffer.seek(0)
        silhouette_plot = base64.b64encode(silhouette_buffer.getvalue()).decode('utf-8')
        plt.close()

        # Perform K-means clustering with selected number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create 2D scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=clusters, cmap=color_scale)
        plt.title('2D Customer Segments')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        cluster_2d_buffer = BytesIO()
        plt.savefig(cluster_2d_buffer, format='png', bbox_inches='tight')
        cluster_2d_buffer.seek(0)
        cluster_2d_plot = base64.b64encode(cluster_2d_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                           c=clusters, cmap=color_scale)
        ax.set_title('3D Customer Segments')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        plt.colorbar(scatter)
        cluster_3d_buffer = BytesIO()
        plt.savefig(cluster_3d_buffer, format='png', bbox_inches='tight')
        cluster_3d_buffer.seek(0)
        cluster_3d_plot = base64.b64encode(cluster_3d_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Calculate cluster statistics
        cluster_stats = []
        for i in range(kmeans.n_clusters):
            cluster_data = data[clusters == i]
            stats = {
                'size': len(cluster_data),
                'percentage': round(len(cluster_data) / len(data) * 100, 2),
                'avg_values': cluster_data[numerical_cols].mean().round(2).to_dict()
            }
            cluster_stats.append(stats)
        
        # After creating kmeans clusters, prepare Plotly data
        plot_data_2d = {
            'data': [{
                'type': 'scatter',
                'mode': 'markers',
                'x': X_pca_2d[:, 0].tolist(),
                'y': X_pca_2d[:, 1].tolist(),
                'marker': {
                    'color': clusters.tolist(),
                    'colorscale': color_scale,
                    'showscale': True,
                    'size': 8
                },
                'text': [f'Cluster {c}' for c in clusters],
                'hoverinfo': 'text'
            }],
            'layout': {
                'title': '2D Customer Segments',
                'xaxis': {'title': 'First Principal Component'},
                'yaxis': {'title': 'Second Principal Component'},
                'showlegend': False,
                'hovermode': 'closest'
            }
        }
        
        plot_data_3d = {
            'data': [{
                'type': 'scatter3d',
                'mode': 'markers',
                'x': X_pca_3d[:, 0].tolist(),
                'y': X_pca_3d[:, 1].tolist(),
                'z': X_pca_3d[:, 2].tolist(),
                'marker': {
                    'color': clusters.tolist(),
                    'colorscale': color_scale,
                    'showscale': True,
                    'size': 5
                },
                'text': [f'Cluster {c}' for c in clusters],
                'hoverinfo': 'text'
            }],
            'layout': {
                'title': '3D Customer Segments',
                'scene': {
                    'xaxis': {'title': 'First Principal Component'},
                    'yaxis': {'title': 'Second Principal Component'},
                    'zaxis': {'title': 'Third Principal Component'}
                },
                'showlegend': False,
                'hovermode': 'closest'
            }
        }
        
        # After creating kmeans clusters, add the cluster labels to the dataframe
        data['Cluster'] = clusters
        
        # Initialize the AI analyzer and get analysis
        analyzer = ClusterAnalyzer(data)
        ai_analysis = analyzer.analyze_clusters()
        
        context = {
            'head_data': head_data,
            'tail_data': tail_data,
            'total_rows': total_rows,
            'total_features': total_features,
            'null_info': null_info,
            'duplicate_info': duplicate_info,
            'description_stats': description_html,
            'cluster_2d_plot': cluster_2d_plot,
            'cluster_3d_plot': cluster_3d_plot,
            'wcss_plot': wcss_plot,
            'silhouette_plot': silhouette_plot,
            'cluster_stats': cluster_stats,
            'total_customers': len(data),
            'features_used': list(numerical_cols),
            'optimal_k': optimal_k,
            'n_clusters': n_clusters,
            'color_scales': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'RdYlBu'],
            'plot_data_2d': json.dumps(plot_data_2d),
            'plot_data_3d': json.dumps(plot_data_3d),
            'ai_analysis': ai_analysis  # Add the AI analysis to the context
        }
        
        return render(request, 'segmentation_result.html', context)
        
    except Exception as e:
        print(f"Error in segmentation_result_view: {str(e)}")
        messages.error(request, f"An error occurred during segmentation: {str(e)}")
        return redirect('segmentation')

def update_clusters(request):
    try:
        if request.method == 'POST':
            # Get parameters from request
            n_clusters = int(request.POST.get('n_clusters', 3))
            color_scale = request.POST.get('color_scale', 'viridis')
            
            # Get dataset from session
            dataset = request.session.get('dataset')
            if not dataset:
                return JsonResponse({'error': 'No dataset found'}, status=400)
            
            # Convert to DataFrame
            data = pd.DataFrame(dataset)
            
            # Get numerical columns
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
            
            # Prepare data for clustering
            X = data[numerical_cols]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            pca_2d = PCA(n_components=2)
            pca_3d = PCA(n_components=3)
            X_pca_2d = pca_2d.fit_transform(X_scaled)
            X_pca_3d = pca_3d.fit_transform(X_scaled)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Prepare 2D plot data
            plot_data_2d = {
                'data': [{
                    'type': 'scatter',
                    'mode': 'markers',
                    'x': X_pca_2d[:, 0].tolist(),
                    'y': X_pca_2d[:, 1].tolist(),
                    'marker': {
                        'color': clusters.tolist(),
                        'colorscale': color_scale,
                        'showscale': True,
                        'size': 8
                    },
                    'text': [f'Cluster {c}' for c in clusters],
                    'hoverinfo': 'text'
                }],
                'layout': {
                    'title': '2D Customer Segments',
                    'xaxis': {'title': 'First Principal Component'},
                    'yaxis': {'title': 'Second Principal Component'},
                    'showlegend': False,
                    'hovermode': 'closest'
                }
            }
            
            # Prepare 3D plot data
            plot_data_3d = {
                'data': [{
                    'type': 'scatter3d',
                    'mode': 'markers',
                    'x': X_pca_3d[:, 0].tolist(),
                    'y': X_pca_3d[:, 1].tolist(),
                    'z': X_pca_3d[:, 2].tolist(),
                    'marker': {
                        'color': clusters.tolist(),
                        'colorscale': color_scale,
                        'showscale': True,
                        'size': 5
                    },
                    'text': [f'Cluster {c}' for c in clusters],
                    'hoverinfo': 'text'
                }],
                'layout': {
                    'title': '3D Customer Segments',
                    'scene': {
                        'xaxis': {'title': 'First Principal Component'},
                        'yaxis': {'title': 'Second Principal Component'},
                        'zaxis': {'title': 'Third Principal Component'}
                    },
                    'showlegend': False,
                    'hovermode': 'closest'
                }
            }
            
            return JsonResponse({
                'plot_data_2d': plot_data_2d,
                'plot_data_3d': plot_data_3d
            })
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def analyze_clusters(request):
    if request.method == 'POST':
        try:
            # Get the DataFrame from the request
            # You'll need to modify this based on how you're sending the data
            df = pd.DataFrame(request.POST.get('data'))
            
            # Initialize the analyzer
            analyzer = ClusterAnalyzer(df)
            
            # Get the analysis
            analysis_result = analyzer.analyze_clusters()
            
            return JsonResponse({
                'status': 'success',
                'analysis': analysis_result
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return render(request, 'cluster_analysis.html')