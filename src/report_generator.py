# # report_generator.py

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib import colors

# def generate_report_md(data_path, report_type="weekly"):
#     # Load your data
#     df = pd.read_csv(data_path)

#     # Process data: Crime trend and location analysis
#     crime_counts = df['crime_types'].value_counts().reset_index()
#     crime_counts.columns = ['Crime Type', 'Count']

#     location_counts = df['locations'].value_counts().reset_index()
#     location_counts.columns = ['Locations', 'Count']
#     location_counts = location_counts.sort_values(by='Count', ascending=False)

#     # Plotting Crime Type Distribution
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x='Count', y='Crime Type', data=crime_counts, ax=ax)
#     crime_plot_path = "crime_type_distribution.png"
#     plt.savefig(crime_plot_path)
#     plt.close()

#     # Plotting Top Crime Locations
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x='Count', y='Locations', data=location_counts.head(10), ax=ax)
#     location_plot_path = "top_crime_locations.png"
#     plt.savefig(location_plot_path)
#     plt.close()

#     # Creating Markdown Content
#     report_content = f"""
#     # {report_type.capitalize()} Crime Analysis Report

#     ## Crime Type Distribution
#     This chart displays the distribution of different crime types reported in the news.

#     ![Crime Type Distribution]({crime_plot_path})

#     ## Top Crime Locations
#     This chart shows the locations with the highest number of reported crimes.

#     ![Top Crime Locations]({location_plot_path})

#     ## Summary
#     - **Total Articles Analyzed**: {len(df)}
#     - **Crime Categories Identified**: {len(crime_counts)}

#     ## Time-based Trends
#     _If available, summarize crime trends over time (weekly/monthly)._

#     **Report Generated on**: {pd.Timestamp.now().strftime('%Y-%m-%d')}
#     """

#     # Save the report as Markdown file
#     report_filename = f"crime_analysis_{report_type}_report.md"
#     with open(report_filename, "w") as file:
#         file.write(report_content)
#     print(f"Markdown report saved as {report_filename}")
#     return report_filename


# def generate_report_pdf(data_path, report_type="weekly"):
#     # Load data and process as done in Markdown report
#     df = pd.read_csv(data_path)
    
#     crime_counts = df['crime_types'].value_counts().reset_index()
#     crime_counts.columns = ['Crime Type', 'Count']

#     location_counts = df['locations'].value_counts().reset_index()
#     location_counts.columns = ['Locations', 'Count']
#     location_counts = location_counts.sort_values(by='Count', ascending=False)

#     # Create plots
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x='Count', y='Crime Type', data=crime_counts, ax=ax)
#     crime_plot_path = "crime_type_distribution.png"
#     plt.savefig(crime_plot_path)
#     plt.close()

#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x='Count', y='Locations', data=location_counts.head(10), ax=ax)
#     location_plot_path = "top_crime_locations.png"
#     plt.savefig(location_plot_path)
#     plt.close()

#     # Create the PDF
#     pdf_filename = f"crime_analysis_{report_type}_report.pdf"
#     c = canvas.Canvas(pdf_filename, pagesize=letter)

#     c.setFont("Helvetica", 16)
#     c.drawString(200, 750, f"{report_type.capitalize()} Crime Analysis Report")
#     c.setFont("Helvetica", 12)

#     # Title section
#     c.drawString(30, 720, "Crime Type Distribution")
#     c.drawImage(crime_plot_path, 30, 500, width=550, height=200)

#     c.drawString(30, 460, "Top Crime Locations")
#     c.drawImage(location_plot_path, 30, 250, width=550, height=200)

#     # Adding Summary Text
#     c.setFont("Helvetica", 10)
#     summary = f"""
#     Total Articles Analyzed: {len(df)}
#     Crime Categories Identified: {len(crime_counts)}
#     Report Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}
#     """
#     c.drawString(30, 230, summary)

#     # Save the PDF
#     c.save()
#     print(f"PDF report saved as {pdf_filename}")
#     return pdf_filename



# report_generator.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os

def generate_report_md(df, output_path=None, report_type="analysis"):
    """Generate markdown report from DataFrame"""
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    
    # Process data
    crime_counts = df['crime_types'].value_counts().reset_index()
    crime_counts.columns = ['Crime Type', 'Count']
    
    location_counts = pd.DataFrame()
    if 'locations' in df.columns:
        location_counts = df['locations'].value_counts().reset_index()
        location_counts.columns = ['Locations', 'Count']
        location_counts = location_counts.sort_values(by='Count', ascending=False)
    
    # Create visualizations
    crime_plot_path = "crime_type_distribution.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Count', y='Crime Type', data=crime_counts, ax=ax)
    plt.savefig(crime_plot_path)
    plt.close()
    
    location_plot_path = None
    if not location_counts.empty:
        location_plot_path = "top_crime_locations.png"
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Count', y='Locations', data=location_counts.head(10), ax=ax)
        plt.savefig(location_plot_path)
        plt.close()
    
    # Create markdown content
    report_content = f"""
# Crime Analysis Report ({report_type.capitalize()})

## Crime Type Distribution
Distribution of different crime types identified in the analysis.

![Crime Type Distribution]({crime_plot_path})
"""
    
    if location_plot_path:
        report_content += f"""
## Top Crime Locations
Locations with the highest number of reported crimes.

![Top Crime Locations]({location_plot_path})
"""
    
    report_content += f"""
## Summary Statistics
- **Total Cases Analyzed**: {len(df)}
- **Crime Categories Identified**: {len(crime_counts)}
- **Unique Locations Found**: {df['locations'].nunique() if 'locations' in df.columns else 0}
"""

    if 'publishedAt' in df.columns:
        report_content += f"""
## Temporal Analysis
- **Date Range**: {df['publishedAt'].min()} to {df['publishedAt'].max()}
"""
    
    report_content += f"""
**Report Generated on**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
    
    # Save or return the report
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            file.write(report_content)
        print(f"Markdown report saved as {output_path}")
        return output_path
    return report_content

def generate_report_pdf(df, output_path=None, report_type="analysis"):
    """Generate PDF report from DataFrame"""
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    
    # Process data
    crime_counts = df['crime_types'].value_counts().reset_index()
    crime_counts.columns = ['Crime Type', 'Count']
    
    location_counts = pd.DataFrame()
    if 'locations' in df.columns:
        location_counts = df['locations'].value_counts().reset_index()
        location_counts.columns = ['Locations', 'Count']
        location_counts = location_counts.sort_values(by='Count', ascending=False)
    
    # Create visualizations
    crime_plot_path = "crime_type_distribution.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Count', y='Crime Type', data=crime_counts, ax=ax)
    plt.savefig(crime_plot_path)
    plt.close()
    
    location_plot_path = None
    if not location_counts.empty:
        location_plot_path = "top_crime_locations.png"
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Count', y='Locations', data=location_counts.head(10), ax=ax)
        plt.savefig(location_plot_path)
        plt.close()
    
    # Create PDF
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        c = canvas.Canvas(output_path, pagesize=letter)
    
    # Title section
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 750, f"Crime Analysis Report ({report_type.capitalize()})")
    c.setFont("Helvetica", 12)
    c.drawString(72, 730, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Crime distribution
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 700, "Crime Type Distribution")
    c.drawImage(crime_plot_path, 72, 450, width=450, height=250)
    
    # Location distribution if available
    if location_plot_path:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, 750, "Top Crime Locations")
        c.drawImage(location_plot_path, 72, 500, width=450, height=250)
    
    # Summary stats
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 750, "Summary Statistics")
    c.setFont("Helvetica", 12)
    
    stats = [
        f"Total Cases Analyzed: {len(df)}",
        f"Crime Categories Identified: {len(crime_counts)}",
        f"Unique Locations Found: {df['locations'].nunique() if 'locations' in df.columns else 0}"
    ]
    
    if 'publishedAt' in df.columns:
        stats.append(f"Date Range: {df['publishedAt'].min()} to {df['publishedAt'].max()}")
    
    y_position = 700
    for stat in stats:
        c.drawString(72, y_position, stat)
        y_position -= 20
    
    c.save()
    print(f"PDF report saved as {output_path}")
    return output_path

def generate_report(df_or_path, output_base_path, report_type="analysis"):
    """Generate both PDF and Markdown reports"""
    # Ensure output directory exists
    os.makedirs(output_base_path, exist_ok=True)
    
    # Generate reports
    pdf_path = os.path.join(output_base_path, f"crime_report_{report_type}.pdf")
    md_path = os.path.join(output_base_path, f"crime_report_{report_type}.md")
    
    generate_report_pdf(df_or_path, pdf_path, report_type)
    generate_report_md(df_or_path, md_path, report_type)
    
    return {
        'pdf': pdf_path,
        'markdown': md_path
    }


