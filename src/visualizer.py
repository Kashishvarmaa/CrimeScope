import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from keyword_cloud import generate_keyword_cloud
from report_generator import generate_report_md, generate_report_pdf,generate_report
from file_processor import extract_text_from_file, process_uploaded_file, generate_analysis_artifacts
import base64
from io import StringIO
import tempfile
import os


# Sidebar Navigation
st.sidebar.title("Crime Analysis Portal")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Navigation", 
                           ["Visualization", "Crime Analysis", "Chatbot"])

# User Info Section
st.sidebar.markdown("---")
st.sidebar.header("User Profile")
user_name = st.sidebar.text_input("Name", "John Doe")
user_role = st.sidebar.selectbox("Role", ["Analyst", "Officer", "Researcher"])
st.sidebar.markdown(f"""
**Current User:** {user_name}  
**Role:** {user_role}  
**Last Login:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
""")


def show_visualization():
    """Current visualization dashboard"""
    df = pd.read_csv("data/ner_output.csv")

    st.title("🕵️‍♀️ Crime Pattern Visualization")
    st.markdown("Interactive visualizations of crime patterns from news reports.")
  

    if st.checkbox("Show raw data"):
        st.dataframe(df.head(10))


    # Drop rows with missing values
    df = df.dropna(subset=['crime_types', 'locations'])

    # --- Crime Trends ---
    st.header("🔍 Crime Type Distribution")
    crime_counts = df['crime_types'].value_counts().reset_index()
    crime_counts.columns = ['Crime Type', 'Count']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Count', y='Crime Type', data=crime_counts, ax=ax)
    st.pyplot(fig)

    # --- Crime Location Heatmap ---
    st.header("📍 Top Crime Locations")
    location_counts = df['locations'].value_counts().reset_index()
    location_counts.columns = ['Locations', 'Count']
    location_counts = location_counts.sort_values(by='Count', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Count', y='Locations', data=location_counts.head(10), ax=ax)
    st.pyplot(fig)

  # --- Keyword Cloud ---
    st.header("☁️ Crime Keyword Cloud")
    wordcloud_fig = generate_keyword_cloud(df)
    st.pyplot(wordcloud_fig)
    
    # Add download button
    if st.button("Download Word Cloud"):
        img_path = "crime_keyword_cloud.png"
        wordcloud_fig.savefig(img_path)
        with open(img_path, "rb") as file:
            st.download_button(
                label="Download as PNG",
                data=file,
                file_name="crime_keyword_cloud.png",
                mime="image/png"
            )


    # New Report Download Section
    st.header("📊 Download Weekly Reports")
    
    if st.button("Generate Weekly Reports"):
        with st.spinner("Generating weekly reports..."):
            report_paths = generate_report(
                "data/geo_news.csv",
                "reports/weekly",
                "weekly"
            )
            
            # Show download buttons
            with open(report_paths['pdf'], "rb") as f:
                st.download_button(
                    "Download PDF Report",
                    f.read(),
                    "weekly_crime_report.pdf",
                    "application/pdf"
                )
            
            with open(report_paths['markdown'], "rb") as f:
                st.download_button(
                    "Download Markdown Report",
                    f.read(),
                    "weekly_crime_report.md",
                    "text/markdown"
                )
        
    # with col1:
    #     if st.button("Generate PDF Report"):
    #         with st.spinner("Generating PDF report..."):
    #             report_path = generate_report_pdf("data/geo_news.csv")
    #             with open(report_path, "rb") as f:
    #                 bytes_data = f.read()
    #             st.download_button(
    #                 label="Download PDF Report",
    #                 data=bytes_data,
    #                 file_name="crime_analysis_report.pdf",
    #                 mime="application/pdf"
    #             )
        
    # with col2:
    #     if st.button("Generate Markdown Report"):
    #         with st.spinner("Generating Markdown report..."):
    #             report_path = generate_report_md("data/geo_news.csv")
    #             with open(report_path, "rb") as f:
    #                 bytes_data = f.read()
    #             st.download_button(
    #                 label="Download Markdown Report",
    #                 data=bytes_data,
    #                 file_name="crime_analysis_report.md",
    #                 mime="text/markdown"
    #             ) 

        

# def plot_heatmap():
#     # Load the geocoded data
#     df = pd.read_csv("data/geo_news.csv")
    
#     # Drop rows with missing lat/lon
#     df = df.dropna(subset=['latitude', 'longitude'])

#     # Ensure values are numeric
#     df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
#     df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
#     df = df.dropna(subset=['latitude', 'longitude'])

#     # Create map
#     m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)

#     # Prepare heatmap data
#     heat_data = df[['latitude', 'longitude']].values.tolist()
#     HeatMap(heat_data).add_to(m)

#     # Display in Streamlit
#     st.header("🔥 Crime Heatmap")
#     folium_static(m)



def plot_heatmap(df=None):
    """Generate heatmap with NaN handling"""
    try:
        if df is None:
            df = pd.read_csv("data/geo_news.csv")
        
        # Clean and validate coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        if len(df) == 0:
            st.warning("No valid location data available for heatmap")
            return None
        
        # Create map centered on mean coordinates
        mean_lat = df['latitude'].mean()
        mean_lon = df['longitude'].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)
        
        # Prepare heatmap data
        heat_data = df[['latitude', 'longitude']].values.tolist()
        HeatMap(heat_data).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"Could not generate heatmap: {str(e)}")
        return None


def show_crime_analysis():
    """File upload and analysis section"""
    st.title("🔬 Crime File Analysis")
    st.markdown("Upload crime reports for detailed analysis.")
    
    CRIME_KEYWORDS = {
        'assault': ['assault', 'attack', 'battery'],
        'theft': ['theft', 'robbery', 'burglary'],
        'murder': ['murder', 'homicide', 'kill'],
        'fraud': ['fraud', 'scam', 'cheat'],
        'sexual': ['rape', 'molest', 'abuse']
    }
    
    uploaded_file = st.file_uploader("Choose a file (PDF or TXT)", 
                                   type=['pdf', 'txt'])
    
    if uploaded_file is not None:
        with st.spinner("Analyzing file. This may take a minute..."):
            try:
                # Process the file through complete pipeline
                df = process_uploaded_file(uploaded_file, CRIME_KEYWORDS)
                artifacts = generate_analysis_artifacts(df)
                
                # Display results
                st.success("Analysis complete!")
                
                # Show summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Crimes Identified", len(df))
                with col2:
                    st.metric("Unique Crime Types", df['crime_types'].nunique())
                with col3:
                    st.metric("Locations Found", df['locations'].nunique())
                
                # Visualization tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Crime Breakdown", "Locations", "Keyword Cloud", "Reports"])
                
                with tab1:
                    st.header("Crime Type Distribution")
                    # fig, ax = plt.subplots()
                    # df['crime_types'].value_counts().plot(kind='bar', ax=ax)
                    # st.pyplot(fig)
                    
                    # st.header("Trend Analysis")
                    # st.image(artifacts['trend_analysis'])
    

                    # Split multi-label crimes and count
                    crime_counts = df['crime_type'].str.split(', ').explode().value_counts()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    crime_counts.plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title("Crime Type Distribution")
                    ax.set_xlabel("Crime Type")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

    
                with tab2:
                    st.header("Crime Locations")
                    
                    # Check if we have location data
                    has_geo_data = 'latitude' in df.columns and 'longitude' in df.columns
                    valid_geo_data = False
                    
                    if has_geo_data:
                        # Clean coordinate data
                        geo_df = df.dropna(subset=['latitude', 'longitude']).copy()
                        geo_df['latitude'] = pd.to_numeric(geo_df['latitude'], errors='coerce')
                        geo_df['longitude'] = pd.to_numeric(geo_df['longitude'], errors='coerce')
                        geo_df = geo_df.dropna(subset=['latitude', 'longitude'])
                        valid_geo_data = len(geo_df) > 0
                    
                    if valid_geo_data:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Map View")
                            try:
                                # Create map centered on first valid location
                                first_valid = geo_df.iloc[0]
                                m = folium.Map(
                                    location=[first_valid['latitude'], first_valid['longitude']], 
                                    zoom_start=10
                                )
                                
                                # Add markers for all locations
                                for _, row in geo_df.iterrows():
                                    folium.Marker(
                                        [row['latitude'], row['longitude']],
                                        popup=f"{row.get('crime_type', 'Unknown')}: {row.get('locations', 'Unknown')}"
                                    ).add_to(m)
                                
                                folium_static(m)
                            except Exception as e:
                                st.error(f"Could not generate map: {str(e)}")
                        
                        with col2:
                            st.subheader("Heatmap View")
                            heatmap = plot_heatmap(geo_df)
                            if heatmap:
                                folium_static(heatmap)
                            else:
                                st.warning("Could not generate heatmap")
                    else:
                        st.warning("No valid geographic coordinates found in the data")
                    
                    # Top locations list (works even without coordinates)
                    st.subheader("Top Locations")
                    if 'locations' in df.columns:
                        location_counts = df['locations'].value_counts().reset_index()
                        fig, ax = plt.subplots()
                        sns.barplot(
                            x='count', 
                            y='locations', 
                            data=location_counts.head(10).rename(columns={'index': 'locations'}),
                            ax=ax
                        )
                        st.pyplot(fig)
                    else:
                        st.warning("No location data available")
              
                
                with tab3:
                    st.header("Keyword Cloud")
                    st.image(artifacts['wordcloud'])
                    with open(artifacts['wordcloud'], "rb") as f:
                        st.download_button(
                            "Download Word Cloud",
                            f.read(),
                            "crime_keyword_cloud.png",
                            "image/png"
                        )
                
                with tab4:
                    st.header("Download Reports")
                    with open(artifacts['pdf_report'], "rb") as f:
                        st.download_button(
                            "Download PDF Report",
                            f.read(),
                            "crime_analysis_report.pdf",
                            "application/pdf"
                        )
                    
                    with open(artifacts['md_report'], "rb") as f:
                        st.download_button(
                            "Download Markdown Report",
                            f.read(),
                            "crime_analysis_report.md",
                            "text/markdown"
                        )
                    
                    st.download_button(
                        "Download Analysis Data (CSV)",
                        df.to_csv(index=False).encode('utf-8'),
                        "crime_analysis_data.csv",
                        "text/csv"
                    )


    # Generate reports for uploaded file
                report_paths = generate_report(
                    df,
                    "data/upload_analysis",
                    "file_analysis"
                )
                
                # Show download buttons
                col1, col2 = st.columns(2)
                with col1:
                    with open(report_paths['pdf'], "rb") as f:
                        st.download_button(
                            "Download PDF Analysis",
                            f.read(),
                            "uploaded_file_analysis.pdf",
                            "application/pdf"
                        )
                with col2:
                    with open(report_paths['markdown'], "rb") as f:
                        st.download_button(
                            "Download Markdown Analysis",
                            f.read(),
                            "uploaded_file_analysis.md",
                            "text/markdown"
                        )


            
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)


def show_chatbot():
    """Placeholder for chatbot interface"""
    st.title("💬 Crime Analysis Chatbot")
    st.markdown("""
    This will be the chatbot interface for querying crime data.
    
    **Coming Soon Features:**
    - Natural language queries about crime patterns
    - Ask for specific crime statistics
    - Get predictive insights
    """)
    
    # Placeholder chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about crime patterns..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Simulate response
        response = f"I'm a placeholder response to: {prompt}"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Main app logic
if app_mode == "Visualization":
    show_visualization()
    plot_heatmap()
elif app_mode == "Crime Analysis":
    show_crime_analysis()
elif app_mode == "Chatbot":
    show_chatbot()



