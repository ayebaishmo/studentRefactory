#!/usr/bin/env python
# coding: utf-8



from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash import html, dcc, Input, Output, State, callback
import google.generativeai as genai
import joblib
import numpy as np
import shap
import os




# Load data
import pandas as pd 
demographic_df = pd.read_csv('data/demographics.csv')
academic_df = pd.read_csv('data/academicPerformance.csv')
activities_df = pd.read_csv('data/extracurricularActivities.csv')
behavior_df = pd.read_csv('data/behavioralPatterns.csv')




demographic_df.head()




academic_df




activities_df



behavior_df




# Rename ID columns to StudentID for consistency
demographic_df.rename(columns={'ID': 'StudentID'}, inplace=True)
academic_df.rename(columns={'Student ID': 'StudentID'}, inplace=True)




merged_df = demographic_df.merge(academic_df, on='StudentID') \
                          .merge(activities_df, on='StudentID') \
                          .merge(behavior_df, on='StudentID')



pd.set_option('display.max_columns', None)
merged_df.head()




merged_df.shape




print("Duplicated rows:",merged_df.duplicated().sum())




missing_representations = ['NA', 'N/A', '', 'na', 'n/a', 'NaN']
missing_check = merged_df.isin(missing_representations) | merged_df.isnull()
missing_summary = missing_check.sum().sort_values(ascending=False)
print("Missing values per column (including text forms):\n", missing_summary[missing_summary > 0])




cat_columns = merged_df.select_dtypes(include="object").columns
print("\n🔤 Categorical column unique values:")
for col in cat_columns:
    print(f"- {col}: {merged_df[col].nunique()} unique values")



# Student with missing grades 
merged_df['Missing grades'] = merged_df[['Python', 'HCD', 'Communication']].isnull().any(axis=1)



# How many student missing grades for each course
merged_df[['Python', 'HCD', 'Communication']].isnull().sum()



# Students who missed all the tests 
merged_df[merged_df[['Python', 'HCD', 'Communication']].isnull().all(axis=1)]


# #### Problem statement
# Refactory seeks to identify key patterns in student academic performance and engagement behaviors to better target support services, improve course offerings, and reduce barriers to student success

# ##### Specifically, the institution aims to:
# - Detect students at risk of poor academic outcomes or dropping out.
# - Understand which behavioral factors (e.g., attendance, participation, punctuality) most strongly relate to performance.
# - Identify gaps in engagement that signal unmet needs in teaching methods, course content, or support services.

# ##### To break the problem down into measurable questions:
# 
# 1. Academic Performance
# - What are the trends in student grades across courses?
# - Are there sudents consistently missing assessments (i.e., NaN grades)?
# - How do test scores relate to student demographics?
# 
# 2. Engagement & Behavior
# - Which behavioral metrics (attendance, participation, punctuality) are most correlated with academic success?
# - Are there students with low engagement but high grades (or vice versa)?
# 
# 3. Experience Gaps
# - Do certain groups (e.g., gender, employment status, marital status) face more challenges in performance or engagement?
# - Are there clusters of students with similar struggles that could benefit from targeted interventions?

# 1. Academic Performance Insights
# - Grade Distribution Analysis
# - Cross-course performance patterns: Compare grade distributions across Javascript, Python, HCD, and Communication courses
# - Grade consistency: Identify students with high variance across subjects vs. consistent performers
# - Missing assessment detection: Track N/A grades to identify students at risk of dropping out
# - Completion rate correlation: Analyze relationship between individual course grades and overall completion status



grade_map = {'A':4.0, 'A-':3.7, 'B+':3.3, 'B':3.0, 'B-':2.7, 'C+':2.3, 'C':2.0, 'N/A': None}
grade_cols = ['Javascript', 'Python', 'HCD', 'Communication']
for col in grade_cols:
    merged_df[col + '_num'] = merged_df[col].map(grade_map)



# Melt for some visuals
melted = merged_df.melt(id_vars=['StudentID', 'Course Completion'], 
                 value_vars=[col + '_num' for col in grade_cols], 
                 var_name='Course', value_name='Grade')
melted['Course'] = melted['Course'].str.replace('_num', '')




def GradeBoxplot(melted_df):
    """
    Creates a styled boxplot figure for grade distribution across courses.
    melted_df: DataFrame with columns 'Course', 'Grade'
    """
    # Define your custom colors for courses
    custom_colors = ['#55c3c7', '#744674', '#684c64']

    fig_box = px.box(
        melted_df,
        x='Course',
        y='Grade',
        color='Course',
        color_discrete_sequence=custom_colors
    )

    fig_box.update_layout(
        title='Grade Distribution Across Courses',
        plot_bgcolor='white',
        paper_bgcolor='#f9f9f9',
        font=dict(
            family='Arial',
            size=14,
            color='black'
        ),
        xaxis=dict(
            title='Course',
            tickfont=dict(color='black'),
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            title='Grade',
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        ),
        legend=dict(
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1,
            font=dict(color='black')
        )
    )

    return html.Div(
        className="chart-card",  # You can style the container with CSS if needed
        children=[
            dcc.Graph(
                figure=fig_box,
                config={'displayModeBar': False}
            )
        ]
    )




# Grade variance per student
merged_df['Grade Std Dev'] = merged_df[[col + '_num' for col in grade_cols]].std(axis=1)
fig_std = px.histogram(
    merged_df,
    x='Grade Std Dev',
    nbins=10,
    title='Grade Consistency (Std Dev) per Student',
    color_discrete_sequence=['#744674'] 
)

# Apply custom layout styling
fig_std.update_layout(
    plot_bgcolor='#f7f7f7', 
    paper_bgcolor='#f9f9f9', 
    font=dict(
        family="Arial",
        size=14,
        color="#333333"
    ),
    title_font=dict(
        size=20,
        color="#2c3e50"
    ),
    xaxis=dict(
        title='Standard Deviation of Grades',
        gridcolor='#e0e0e0',
        linecolor='#2c3e50',
        zerolinecolor='#cccccc'
    ),
    yaxis=dict(
        title='Number of Students',
        gridcolor='#e0e0e0',
        linecolor='#2c3e50',
        zerolinecolor='#cccccc'
    ),
    bargap=0.1,
)



# Count missing grades per student
merged_df['Missing Grades'] = merged_df[[col + '_num' for col in grade_cols]].isna().sum(axis=1)

# Create histogram with custom color
fig_missing = px.histogram(
    merged_df,
    x='Missing Grades',
    title='Number of Missing Grades per Student',
    color_discrete_sequence=['#55c3c7']
)

# Apply custom styling
fig_missing.update_layout(
    plot_bgcolor='#f7f7f7',
    paper_bgcolor='#ffffff', 
    font=dict(
        family="Arial",
        size=14,
        color="#333333"
    ),
    title_font=dict(
        size=20,
        color="#2c3e50"
    ),
    xaxis=dict(
        title='Missing Grade Count',
        gridcolor='#e0e0e0',
        linecolor='#2c3e50',
        zerolinecolor='#cccccc'
    ),
    yaxis=dict(
        title='Number of Students',
        gridcolor='#e0e0e0',
        linecolor='#2c3e50',
        zerolinecolor='#cccccc'
    ),
    bargap=0.1,
)




# Completion status comparison
fig_violin = px.violin(
    melted,
    x='Course',
    y='Grade',
    color='Course Completion',  # categorical color
    box=True,
    points='all',
    title='Grade Comparison by Course Completion Status',
    color_discrete_sequence=['#744674', '#55c3c7']  # customize for each category
)

# Apply custom layout styling
fig_violin.update_layout(
    plot_bgcolor='#f7f7f7',       # Inside the plot area
    paper_bgcolor='#ffffff',      # Outside the plot area
    font=dict(
        family="Arial",
        size=14,
        color="#333333"
    ),
    title_font=dict(
        size=20,
        color="#2c3e50"
    ),
    xaxis=dict(
        title='Course',
        gridcolor='#e0e0e0',
        linecolor='#2c3e50',
        zerolinecolor='#cccccc'
    ),
    yaxis=dict(
        title='Grade',
        gridcolor='#e0e0e0',
        linecolor='#2c3e50',
        zerolinecolor='#cccccc'
    )
)


# Behavioral



# correlation between time spent on materials and academic performance.
# Calculate average grade
merged_df['Average Grade'] = merged_df[[col + '_num' for col in ['Javascript', 'Python', 'HCD', 'Communication']]].mean(axis=1)

# Create scatter plot
fig_time = px.scatter(
    merged_df,
    x='Time Spent On Materials (Hours)',
    y='Average Grade',
    trendline='ols',
    title='Study Time vs Academic Performance',
    hover_data=['StudentID'],
    color_discrete_sequence=['#1f77b4']  # Optional: single point color
)

# Customize layout (backgrounds, gridlines, etc.)
fig_time.update_layout(
    paper_bgcolor='#f8f9fa',   # Full background (outside plot)
    plot_bgcolor='#ffffff',    # Inside plot area background
    title_font_color='black',
    font=dict(color='black'),
    xaxis=dict(
        title='Time Spent On Materials (Hours)',
        gridcolor='#e0e0e0',
        linecolor='black',
        zerolinecolor='black'
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#e0e0e0',
        linecolor='black',
        zerolinecolor='black'
    )
)

# Customize marker style if needed
fig_time.update_traces(marker=dict(size=10, color='#55c3c7'))

# Show figure




# Forum Engagement Impact
# Understand if forum participation (posts + time) leads to better grades. 
fig_forum = px.scatter(
    merged_df,
    x='Forum Posts',
    y='Average Grade',
    size='Time Spent On Forum (Hours)',
    color='Course Completion',  # This will automatically get a color scale
    title='Forum Engagement vs Academic Performance',
    hover_data=['StudentID'],
    color_discrete_sequence=['#55c3c7', '#744674']  # customize if categorical
)

# Customize layout (backgrounds, fonts, axes)
fig_forum.update_layout(
    paper_bgcolor='#f0f0f0',   # Full figure background
    plot_bgcolor='#ffffff',    # Plot area background
    title_font_color='black',
    font=dict(color='black', size=14),
    xaxis=dict(
        title='Forum Posts',
        gridcolor='#dcdcdc',
        linecolor='black',
        zerolinecolor='black',
        # tickfont=dict(color='black'),
        # titlefont=dict(color='black')
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#dcdcdc',
        linecolor='black',
        zerolinecolor='black',
        # tickfont=dict(color='black'),
        # titlefont=dict(color='black')
    ),
    legend=dict(
        bgcolor='#f0f0f0',
        bordercolor='gray',
        borderwidth=1,
        font=dict(color='black')
    )
)

# Optional: Change marker styling globally
fig_forum.update_traces(marker=dict(line=dict(width=1, color='black')))

# Show the figure




# Communication Patterns (Instructor Messages)
# Analyze how communication with instructors relates to grades.
fig_msgs = px.scatter(
    merged_df,
    x='Instructor Messages',
    y='Average Grade',
    title='Instructor Messages vs Academic Performance',
    hover_data=['StudentID'],
    color_discrete_sequence=['#744674']  # Customize point color here
)

fig_msgs.update_layout(
    paper_bgcolor='#fafafa',   # Figure background (outside plot)
    plot_bgcolor='#ffffff',    # Plot area background
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    xaxis=dict(
        title='Instructor Messages',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653')
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653')
    )
)

fig_msgs.update_traces(marker=dict(size=10, line=dict(width=1, color='#264653')))





# Assignment Completion Patterns 
# Show patterns between assignment completion and grades. 
fig_assign = px.scatter(
    merged_df,
    x='Completed Assignments',
    y='Average Grade',
    title='Assignment Completion vs Academic Performance',
    hover_data=['StudentID'],
    color_discrete_sequence=['#2a9d8f']  # Customize marker color here
)

fig_assign.update_layout(
    paper_bgcolor='#f7f9f9',   # Figure background (outside plot)
    plot_bgcolor='#ffffff',    # Plot area background
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    xaxis=dict(
        title='Completed Assignments',
        gridcolor='#d3d3d3',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653')
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#d3d3d3',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653')
    )
)

fig_assign.update_traces(marker=dict(size=9, line=dict(width=1, color='#264653')))



# Performance Gap Identification



# how demographic groups relate to academic performance (average or per-course grades).
fig_gender = px.box(
    merged_df,
    x='Gender',
    y='Average Grade',
    title='Gender vs Academic Performance',
    color='Gender',  # color boxes by gender
    color_discrete_map={'Male': '#55c3c7', 'Female': '#643464'},  # customize colors
)

fig_gender.update_layout(
    paper_bgcolor='#f9fafb',   # Figure background
    plot_bgcolor='#ffffff',    # Plot area background
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    xaxis=dict(
        title='Gender',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653')
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653')
    ),
    legend=dict(
        title='Gender',
        bgcolor='#f9fafb',
        bordercolor='#ccc',
        borderwidth=1,
        font=dict(color='#264653')
    )
)

fig_gender.update_traces(
    marker=dict(line=dict(width=1, color='#264653'))  # box outline color
)





# Income scatter
fig_income = px.scatter(
    merged_df,
    x='Income Level',
    y='Average Grade',
    trendline='ols',
    title='Income vs Academic Performance',
    color_discrete_sequence=['#643464']  # Customize marker color here (blueviolet)
)

fig_income.update_layout(
    paper_bgcolor='#f5f7fa',   # Figure background (outside plot)
    plot_bgcolor='#ffffff',    # Plot area background
    title_font_color='#2c3e50',
    font=dict(color='#2c3e50', size=14),
    xaxis=dict(
        title='Income Level',
        gridcolor='#dcdcdc',
        linecolor='#2c3e50',
        zerolinecolor='#2c3e50',
        # tickfont=dict(color='#2c3e50'),
        # titlefont=dict(color='#2c3e50')
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#dcdcdc',
        linecolor='#2c3e50',
        zerolinecolor='#2c3e50',
        # tickfont=dict(color='#2c3e50'),
        # titlefont=dict(color='#2c3e50')
    )
)

fig_income.update_traces(marker=dict(size=10, line=dict(width=1, color='#2c3e50')))





# Employment Impact 
fig_employment = px.box(
    merged_df,
    x='Employment Status',
    y='Average Grade',
    title='Employment Status vs Academic Performance',
    color='Employment Status',
    color_discrete_map={
        'Full-time': '#2a9d8f',
        'Part-time': '#643464',
        'Unemployed': '#e3dde5',
        # Add more categories/colors as needed
    }
)

fig_employment.update_layout(
    paper_bgcolor='#f9fbfc',    # Figure background
    plot_bgcolor='#ffffff',     # Plot area background
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    xaxis=dict(
        title='Employment Status',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    legend=dict(
        title='Employment Status',
        bgcolor='#f9fbfc',
        bordercolor='#ccc',
        borderwidth=1,
        font=dict(color='#264653')
    )
)

fig_employment.update_traces(
    marker=dict(line=dict(width=1, color='#264653'))  # box outlines
)




# Geographic Factors 
district_avg = merged_df.groupby("District")["Average Grade"].mean().reset_index()

my_colors = {
    'Fort Portal': '#e3dde5',  # blue
    'Gulu': '#55c3c7',  # orange
    'Kampala': '#744674',  # green
    'Mukono': '#a181a1',  # red
    'Wakiso': '#55c3c7',  # purple
}

fig_district = px.bar(
    district_avg,
    x='District',
    y='Average Grade',
    title='Average Grade by District',
    color='District',
    color_discrete_map=my_colors  # Use your custom colors here
)

fig_district.update_layout(
    paper_bgcolor='#f9fafb',
    plot_bgcolor='#ffffff',
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    xaxis=dict(
        title='District',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    legend=dict(
        title='District',
        bgcolor='#f9fafb',
        bordercolor='#ccc',
        borderwidth=1,
        font=dict(color='#264653')
    )
)

fig_district.update_traces(marker=dict(line=dict(width=1, color='#264653')))




# Family Responsibility (Number of Children)
fig_kids = px.scatter(
    merged_df,
    x='Number Of Children',
    y='Average Grade',
    trendline='ols',
    title='Family Responsibility vs Academic Performance',
    color_discrete_sequence=['#55c3c7']  # Customize marker color (orange sandy)
)

fig_kids.update_layout(
    paper_bgcolor='#fbfbfb',   # Figure background
    plot_bgcolor='#ffffff',    # Plot area background
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    xaxis=dict(
        title='Number Of Children',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653')
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653')
    )
)

fig_kids.update_traces(marker=dict(size=10, line=dict(width=1, color='#264653')))





#  Marital Status Impact 
fig_marital = px.box(
    merged_df,
    x='Marital Status',
    y='Average Grade',
    title='Marital Status vs Academic Performance',
    # color='Marital Status',
    color_discrete_map={
        'Single': '#2a9d8f',
        'Married': '#e76f51',
        'Divorced': '#264653',
        'Widowed': '#f4a261',
        # Add more categories if needed
    }
)

fig_marital.update_layout(
    paper_bgcolor='#f9fafb',    # Figure background
    plot_bgcolor='#ffffff',     # Plot area background
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    xaxis=dict(
        title='Marital Status',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    legend=dict(
        title='Marital Status',
        bgcolor='#f9fafb',
        bordercolor='#ccc',
        borderwidth=1,
        font=dict(color='#264653')
    )
)

fig_marital.update_traces(
    marker=dict(line=dict(width=1, color='#264653'))  # box outlines
)





# Education Level vs Performance 
edu_avg = merged_df.groupby("Education Level")["Average Grade"].mean().reset_index()

# Define your custom colors per education level
edu_colors = {
    'High School': '#744674',
    'Undergraduate': '#2a9d8f',
    'Postgraduate': '#e3dde5',
    'Doctorate': '#684c64',
    # Add more levels/colors as needed
}
# 
fig_edu = px.bar(
    edu_avg,
    x='Education Level',
    y='Average Grade',
    title='Education Level vs Academic Performance',
    color='Education Level',
    color_discrete_map=edu_colors
)

fig_edu.update_layout(
    paper_bgcolor='#f9fafb',   # Figure background
    plot_bgcolor='#ffffff',    # Plot area background
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    xaxis=dict(
        title='Education Level',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    legend=dict(
        title='Education Level',
        bgcolor='#f9fafb',
        bordercolor='#ccc',
        borderwidth=1,
        font=dict(color='#264653')
    )
)

fig_edu.update_traces(marker=dict(line=dict(width=1, color='#264653')))





#  Location-Based Resource Access 
location_colors = {
    'Urban': '#2a9d8f',
    'Suburban': '#744674',
    'Rural': '#bca4bc',
}

fig_location_study = px.box(
    merged_df,
    x='Location',
    y='Time Spent On Materials (Hours)',
    title='Location vs Study Time',
    color='Location',
    color_discrete_map=location_colors
)

fig_location_study.update_layout(
    paper_bgcolor='#f9fafb',   # Figure background
    plot_bgcolor='#ffffff',    # Plot area background
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    xaxis=dict(
        title='Location',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    yaxis=dict(
        title='Time Spent On Materials (Hours)',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    legend=dict(
        title='Location',
        bgcolor='#f9fafb',
        bordercolor='#ccc',
        borderwidth=1,
        font=dict(color='#264653')
    )
)

fig_location_study.update_traces(
    marker=dict(line=dict(width=1, color='#264653'))
)



# Extracurricular Activity Insights



# Pie of participation
participation_colors = {
    'Active': '#2a9d8f',
    'Inactive': '#744674',
    # Add more categories/colors as needed
}

fig_participation_pie = px.pie(
    merged_df,
    names='Participation Status',
    title='Participation in Extracurricular Activities',
    color='Participation Status',
    color_discrete_map=participation_colors
)

fig_participation_pie.update_traces(
    textposition='inside',
    textinfo='percent+label',
    marker=dict(line=dict(color='#264653', width=1))
)

fig_participation_pie.update_layout(
    paper_bgcolor='#f9fafb',
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    legend=dict(
        bgcolor='#f9fafb',
        bordercolor='#ccc',
        borderwidth=1,
        font=dict(color='#264653')
    )
)


# Bar of average performance
participation_perf = merged_df.groupby("Participation Status")["Average Grade"].mean().reset_index()

fig_participation_perf = px.bar(
    participation_perf,
    x='Participation Status',
    y='Average Grade',
    title='Academic Performance by Participation',
    color='Participation Status',
    color_discrete_map=participation_colors
)

fig_participation_perf.update_layout(
    paper_bgcolor='#f9fafb',
    plot_bgcolor='#ffffff',
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    xaxis=dict(
        title='Participation Status',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653'),
    ),
    legend=dict(
        title='Participation Status',
        bgcolor='#f9fafb',
        bordercolor='#ccc',
        borderwidth=1,
        font=dict(color='#264653')
    )
)

fig_participation_perf.update_traces(marker=dict(line=dict(width=1, color='#264653')))





# Leadership Roles Correlation 
fig_leadership = px.bar(
    merged_df.groupby("Role")["Average Grade"].mean().reset_index(),
    x="Role", y="Average Grade",
    title="Academic Performance by Leadership Role",
    color="Average Grade",
    color_continuous_scale="teal"
)


# Melt for radar
radar_data = merged_df.groupby("Role")[["Average Grade", "Forum Posts", "Completed Assignments", "Time Spent On Materials (Hours)"]].mean().reset_index()

# Melt it into long format for radar
radar_df = radar_data.melt(id_vars="Role", var_name="Metric", value_name="Value")

# Plot radar chart
fig_radar = px.line_polar(
    radar_df,
    r="Value",
    theta="Metric",
    color="Role",
    line_close=True,
    title="Academic & Engagement Radar by Role",

)

fig_radar.update_traces(fill='toself')  # Optional for filled radar look




# Time Management (Hours in Activities vs. Performance) 

fig_violin = px.violin(
    merged_df,
    y='Average Grade',
    x='Hours Per Week',  
    box=True,
    points='all',
    title='Performance by Activity Involvement Level',
    color_discrete_sequence=['#2a9d8f', '#683464' ]  # deep teal
)

fig_violin.update_layout(
    paper_bgcolor='#f9fafb',
    plot_bgcolor='#ffffff',
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    xaxis=dict(
        title='Hours Per Week',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653')
    ),
    yaxis=dict(
        title='Average Grade',
        gridcolor='#e0e0e0',
        linecolor='#264653',
        zerolinecolor='#264653',
        # tickfont=dict(color='#264653'),
        # titlefont=dict(color='#264653')
    ),
    legend=dict(
        bgcolor='#f9fafb',
        bordercolor='#ccc',
        borderwidth=1,
        font=dict(color='#264653')
    )
)

fig_violin.update_traces(meanline_visible=True,
                         marker=dict(color='rgba(42, 157, 143, 0.3)', line=dict(width=1, color='#264653')))





# Sunburst for nested breakdown (if multiple levels like Activity Type → Role)
custom_colors = {
    'Student Government': '#2a9d8f',
    'Drama Club': '#643464',
    'Debate Club': '#643464',
    'Chess Club': '#4ca4c8',
    'Drama': '#a8dadc',
    'Volunteering': '#4ca4c8',
    'Music Band': '#4c5c64',
    'Art Club': '#264653',
    'Football': '#643464',
}

fig_sunburst = px.sunburst(
    merged_df,
    path=['Activity', 'Role'],
    values='Average Grade',
    color='Activity',
    color_discrete_map=custom_colors,
    title='Activity Type & Leadership Breakdown'
)

fig_sunburst.update_layout(
    paper_bgcolor='#f9fafb',
    plot_bgcolor='#ffffff',
    title_font_color='#264653',
    font=dict(color='#264653', size=14),
    margin=dict(t=50, l=10, r=10, b=10)
)





# Activity vs Engagement Heatmap 
heat_df = merged_df.groupby("Activity")[
    ["Average Grade", "Forum Posts", "Time Spent On Materials (Hours)"]
].mean().reset_index()

# Create the heatmap
fig_heat = px.imshow(
    heat_df.set_index("Activity"),
    text_auto=True,
    aspect="auto",
    title='Engagement Metrics by Activity Type',
    color_continuous_scale='teal'  # Use your own palette, or try 'Viridis', 'Plasma', etc.
)

# Customize layout and styling
fig_heat.update_layout(
    paper_bgcolor='#f9fafb',
    plot_bgcolor='#ffffff',
    title_font_color='#264653',
    font=dict(color='#264653', size=13),
    margin=dict(t=50, l=20, r=20, b=20),
    coloraxis_colorbar=dict(title='Metric Scale')
)





print(merged_df.columns.tolist())


custom_color_map = {
    "Engaged High Achievers": "#2a9d8f",
    "Struggling Engagers": "#643464",
    "At-Risk Students": "#4ca4c8",
    "Natural Talents": "#b494b4",
    "Others": "#e2dce4"
}

# fig_archetype_pie = px.pie(
#     merged_df,
#     names="Archetype",
#     title="Distribution of Student Archetypes",
#     hole=0.4,
#     color="Archetype",  # this tells Plotly which column to map
#     color_discrete_map=custom_color_map
# )

# # Optional: pull out at-risk students to highlight
# fig_archetype_pie.update_traces(
#     textinfo='percent+label',
#     pull=[0.05 if val == "At-Risk Students" else 0 for val in merged_df["Archetype"]]
# )

# Custom background and font
# fig_archetype_pie.update_layout(
#     paper_bgcolor='#f9fafb',
#     title_font_color='#264653',
#     font=dict(color='#264653', size=13)
# )





# Scatter plot for Success Predictors
fig_success = px.scatter(
    merged_df,
    x="Attendance %",
    y="Completed Assignments",
    color="Participation Status",
    size="Income Level",
    hover_name="StudentID",
    title="Success Predictors: Attendance vs. Assignments with Participation and Income",
    color_discrete_map={
        "Active": "#704270",
        "Moderate": "#e2dce4",
        "Low": "#55c3c7"
    }
)



# Course Design Insights



# Count N/As
subjects = ["Javascript", "Python", "HCD", "Communication"]
na_counts = merged_df[subjects].isna().sum().reset_index()
na_counts.columns = ['Subject', 'NA Count']

# Bar chart with custom colors
fig_na = px.bar(
    na_counts,
    x="Subject",
    y="NA Count",
    title="Subjects with Highest N/A (Missing) Grades",
    color="Subject",
    color_discrete_sequence=["#704270", "#e2dce4", "#55c3c7", "#b494b4"]
)

# Set background and font
fig_na.update_layout(
    plot_bgcolor="#f0f0f0",
    paper_bgcolor="#ffffff",
    font=dict(color="#333333", family="Arial", size=14),
    title_font=dict(size=18),
    legend=dict(bgcolor="#ffffff", bordercolor="#cccccc", borderwidth=1)
)





subjects = ["Javascript", "Python", "HCD", "Communication"]

# Convert grades to GPA values
merged_df['Average Grade'] = merged_df[subjects].apply(lambda row: pd.to_numeric(row.map({
    'A': 4, 'B+': 3.5, 'B': 3, 'B-': 2.7, 'C+': 2.5, 'C': 2, 'D': 1, 'N/A': None
}), errors='coerce').mean(), axis=1)

# Create scatter plot
fig_corr = px.scatter(
    merged_df,
    x="Completed Assignments",
    y="Average Grade",
    color="Course Completion",
    size="Time Spent On Materials (Hours)",
    title="Assessment Completion vs Final Grade",
    color_discrete_map={
        "Completed": "#704270",
        "Incomplete": "#55c3c7"
    }
)

# Add custom background colors and fonts
fig_corr.update_layout(
    plot_bgcolor="#f0f0f0",      # Inner plot area
    paper_bgcolor="#ffffff",     # Outer figure area (canvas)
    font=dict(color="#222222", family="Arial", size=14),
    title_font=dict(size=18, color="#333333"),
    legend=dict(
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1
    )
)




fig_util = px.scatter(
    merged_df,
    x="Time Spent On Materials (Hours)",
    y="Average Grade",
    color="Socioeconomic Status",
    title="Study Time vs Grade Performance",
    hover_name="StudentID",
    size="Completed Assignments",
    color_discrete_map={
        "Low": "#704270",
        "Middle": "#e2dce4",
        "High": "#55c3c7"
    }
)

# Add background, font, legend styling
fig_util.update_layout(
    plot_bgcolor="#f0f0f0",       # Inside axes
    paper_bgcolor="#ffffff",      # Outer canvas
    font=dict(color="#222", family="Arial", size=14),
    title_font=dict(size=18, color="#333333"),
    legend=dict(
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1
    )
)




fig_support = px.box(
    merged_df,
    x="Number Of Children",
    y="Average Grade",
    color="Course Completion",
    title="Performance Distribution by Number of Children",
    color_discrete_map={
        "Completed": "#704270",
        "Incomplete": "#55c3c7"
    }
)

# Add background and layout styling
fig_support.update_layout(
    plot_bgcolor="#f0f0f0",       # Plot background
    paper_bgcolor="#ffffff",      # Canvas background
    font=dict(color="#222222", family="Arial", size=14),
    title_font=dict(size=18, color="#333333"),
    legend=dict(
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1
    )
)




fig_access = px.box(
    merged_df,
    x="Location",
    y="Time Spent On Materials (Hours)",
    color="Location",
    title="Study Time by Location (Urban vs Rural)",
    color_discrete_sequence=["#55c3c7", "#744674", "#684c64"]
)

fig_access.update_layout(
    plot_bgcolor="#f0f0f0",       # Inside the plot area
    paper_bgcolor="#ffffff",      # Around the entire figure
    font=dict(color="#222222", family="Arial", size=14),
    title_font=dict(size=18, color="#333333"),
    legend=dict(
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1
    )
)




# Ensure 'Date' column is in datetime format
merged_df['Date'] = pd.to_datetime(merged_df['Date'])

# Group forum posts by date
forum_by_date = merged_df.groupby("Date")["Forum Posts"].sum().reset_index()

# Create the line chart
fig_forum_2 = px.line(
    forum_by_date,
    x="Date",
    y="Forum Posts",
    title="Forum Engagement Over Time",
    markers=True  # Optional: shows dots on line
)

# Add background and layout styling
fig_forum_2.update_layout(
    plot_bgcolor="#f0f0f0",       # Plot area background
    paper_bgcolor="#ffffff",      # Full figure background
    font=dict(color="#222222", family="Arial", size=14),
    title_font=dict(size=18, color="#333333"),
    legend=dict(
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1
    )
)




# How engagement and grades vary across the academic calendar 
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df['Month'] = merged_df['Date'].dt.to_period('M').astype(str)

monthly_avg = merged_df.groupby('Month').agg({
    'Time Spent On Materials (Hours)': 'mean',
    'Average Grade': 'mean'
}).reset_index()

fig_seasonal = px.line(
    monthly_avg,
    x="Month",
    y=["Time Spent On Materials (Hours)", "Average Grade"],
    title="Seasonal Trends: Study Time and Grade Averages",
    markers=True
)

fig_seasonal.update_layout(
    plot_bgcolor="#f0f0f0",
    paper_bgcolor="#ffffff",
    font=dict(color="#222222", family="Arial", size=14),
    title_font=dict(size=18, color="#333333"),
    legend=dict(
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1
    )
)





merged_df["Weekday"] = merged_df["Date"].dt.day_name()

weekday_engagement = merged_df.groupby("Weekday")["Forum Posts"].sum().reindex([
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]).reset_index()

fig_weekly = px.bar(
    weekday_engagement,
    x="Weekday",
    y="Forum Posts",
    title="Forum Engagement by Day of the Week",
    color="Weekday",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig_weekly.update_layout(
    plot_bgcolor="#f0f0f0",
    paper_bgcolor="#ffffff",
    font=dict(color="#222222", family="Arial", size=14),
    title_font=dict(size=18, color="#333333"),
    legend=dict(
        bgcolor="#ffffff",
        bordercolor="#cccccc",
        borderwidth=1
    )
)





performance_over_time = merged_df.groupby("Date")["Average Grade"].mean().reset_index()

fig_progress = px.line(
    performance_over_time,
    x="Date",
    y="Average Grade",
    title="Student Performance Progression Over Time",
    markers=True
)

fig_progress.update_layout(
    plot_bgcolor="#f0f0f0",
    paper_bgcolor="#ffffff",
    font=dict(color="#222222", family="Arial", size=14),
    title_font=dict(size=18, color="#333333")
)





dropouts = merged_df[merged_df["Course Completion"] == "Incomplete"]
dropouts_by_date = dropouts.groupby("Date")["StudentID"].count().reset_index()

fig_dropout = px.bar(
    dropouts_by_date,
    x="Date",
    y="StudentID",
    title="Dropout Frequency Over Time",
    labels={"StudentID": "Dropout Count"},
    color_discrete_sequence=["#d62728"]
)

fig_dropout.update_layout(
    plot_bgcolor="#f0f0f0",
    paper_bgcolor="#ffffff",
    font=dict(color="#222222", family="Arial", size=14),
    title_font=dict(size=18, color="#333333")
)


def create_kpi_cards(merged_df):
    # Mapping all known grades
    grade_map = {
        "A": 4, "A-": 3.7,
        "B+": 3.5, "B": 3, "B-": 2.7,
        "C+": 2.5, "C": 2, "C-": 1.7,
        "D": 1, "F": 0
    }

    # Replace grades with numeric values, unrecognized become NaN
    df_num = merged_df.copy()
    for col in ["Javascript", "Python", "HCD", "Communication"]:
        df_num[col] = df_num[col].map(grade_map).astype(float)

    # Calculate averages safely
    avg_js = round(df_num["Javascript"].mean(skipna=True), 1)
    avg_py = round(df_num["Python"].mean(skipna=True), 1)
    avg_hcd = round(df_num["HCD"].mean(skipna=True), 1)
    avg_comm = round(df_num["Communication"].mean(skipna=True), 1)

    return dbc.Row([
        dbc.Col(dbc.Card( 
            dbc.CardBody([
                html.H6("Avg JavaScript", className="card-title"),
                html.Div([
                    html.Img(src='assets/speed.png', style={'height': '30px', 'marginRight': '10px'}),
                    html.H2(avg_js, className="card-value", style={'margin': 0})
                    ], style={'display': 'flex', 'alignItems': 'center'}), 
                html.Small("average grade", className="card-desc"),
            ], className="CardB")
        ), width=3),

        dbc.Col(dbc.Card(
            dbc.CardBody([
                html.H6("Avg Python", className="card-title"),
                html.Div([
                    html.Img(src='assets/speed.png', style={'height': '30px', 'marginRight': '10px'}),
                    html.H2(avg_py, className="card-value", style={'margin': 0})
                    ], style={'display': 'flex', 'alignItems': 'center'}), 
                html.Small("average grade", className="card-desc"),
            ], className="CardB")
        ), width=3),

        dbc.Col(dbc.Card(
            dbc.CardBody([
                html.H6("Avg HCD", className="card-title"),
                html.Div([
                    html.Img(src='assets/speed.png', style={'height': '30px', 'marginRight': '10px'}),
                    html.H2(avg_hcd, className="card-value", style={'margin': 0})
                    ], style={'display': 'flex', 'alignItems': 'center'}), 
                html.Small("average grade", className="card-desc"),
            ], className="CardB")
        ), width=3),

        dbc.Col(dbc.Card(
            dbc.CardBody([
                html.H6("Avg Communication", className="card-title"),
                html.Div([
                    html.Img(src='assets/speed.png', style={'height': '30px', 'marginRight': '10px'}),
                    html.H2(avg_comm, className="card-value", style={'margin': 0})
                    ], style={'display': 'flex', 'alignItems': 'center'}), 
                html.Small("average grade", className="card-desc"),
            ], className="CardB")
        ), width=3),
    ], className="mb-4")

def GradePieChart(merged_df):
    """
    Generates a styled pie chart for grade distribution.
    df: pandas DataFrame containing grade data
    """
    # Count grade distribution
    grade_counts = merged_df['Javascript'].value_counts().reset_index()
    grade_counts.columns = ['Grade', 'Count']

    custom_colors = ["#e3dde5", '#55c3c7', '#684c64', '#AB63FA', '#FFA15A', '#19D3F3']

    # Create Pie Chart
    fig = px.pie(
        grade_counts,
        names='Grade',
        values='Count',
        title="JavaScript Grade Distribution",
        hole=0.4,
        color_discrete_sequence=custom_colors
    )

    # Force legend to appear
    fig.update_traces(
        textinfo='percent',  # Only % on chart
        hovertemplate='%{label}: %{value} students (%{percent})<extra></extra>'
    )

    fig.update_layout(
        showlegend=True,  # Force legend
        legend_title="Grades",
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="black")
    )

    return html.Div(
        className="chart-card",
        children=[
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': False}
            )
        ]
    )


def create_engagement_cards(merged_df):
    metrics = [
        ("Avg Time on Materials (hrs)", "Time Spent On Materials (Hours)", "assets/materials.png"),
        ("Average Forum Posts", "Forum Posts", "assets/forum.png"),
        ("Avg Instructor Messages", "Instructor Messages", "assets/messages.png"),
        ("Avg Completed Assignments", "Completed Assignments", "assets/assignments.png")
    ]

    cards = []
    for title, col, icon in metrics:
        avg_value = round(merged_df[col].mean(skipna=True), 1)
        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6(title, className="card-title"),
                        html.Div([
                            html.Img(src='assets/speed.png', style={'height': '30px', 'marginRight': '10px'}),
                            html.H2(avg_value, className="card-value", style={'margin': 0})
                        ], style={'display': 'flex', 'alignItems': 'center'}),
                        html.Small("average engagement metric", className="card-desc"),
                    ], className="CardB")
                ),
                width=3
            )
        )

    return dbc.Row(cards, className="mb-4")

def DemographyForm():
    return dbc.Card(
        [
            html.H5("Demography Data", className="card-title mb-3"),

            dbc.Form([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Age"),
                        dbc.Input(type="number", id="age", placeholder="Enter Age", min=0, required=True),
                    ], md=4),

                    dbc.Col([
                        dbc.Label("Marital Status"),
                        dbc.Select(
                            id="marital_status",
                            options=[
                                {"label": "Single", "value": "Single"},
                                {"label": "Married", "value": "Married"},
                                {"label": "Divorced", "value": "Divorced"},
                                {"label": "Widowed", "value": "Widowed"},
                            ],
                            value="Single"
                        ),
                    ], md=4),

                    dbc.Col([
                        dbc.Label("Employment Status"),
                        dbc.Select(
                            id="employment_status",
                            options=[
                                {"label": "Unemployed", "value": "Unemployed"},
                                {"label": "Part-time", "value": "Part-time"},
                                {"label": "Full-time", "value": "Full-time"},
                                {"label": "Self-employed", "value": "Self-employed"},
                            ],
                            value="Unemployed"
                        ),
                    ], md=4),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Gender"),
                        dbc.Select(
                            id="gender_demo",
                            options=[
                                {"label": "Male", "value": "Male"},
                                {"label": "Female", "value": "Female"},
                                {"label": "Other", "value": "Other"},
                            ],
                            value="Male"
                        ),
                    ], md=4),

                    dbc.Col([
                        dbc.Label("Socioeconomic Status"),
                        dbc.Select(
                            id="socioeconomic",
                            options=[
                                {"label": "Low", "value": "Low"},
                                {"label": "Middle", "value": "Middle"},
                                {"label": "High", "value": "High"},
                            ],
                            value="Middle"
                        ),
                    ], md=4),

                    dbc.Col([
                        dbc.Label("Income Level"),
                        dbc.Input(type="number", id="income_level", placeholder="Enter Income", min=0, required=True),
                    ], md=4),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Location"),
                        dbc.Select(
                            id="location",
                            options=[
                                {"label": "Urban", "value": "Urban"},
                                {"label": "Rural", "value": "Rural"},
                            ],
                            value="Urban"
                        ),
                    ], md=4),

                    dbc.Col([
                        dbc.Label("District"),
                        dbc.Input(type="text", id="district_demo", placeholder="Enter District"),
                    ], md=4),

                    dbc.Col([
                        dbc.Label("Education Level"),
                        dbc.Select(
                            id="education",
                            options=[
                                {"label": "Primary", "value": "Primary"},
                                {"label": "Secondary", "value": "Secondary"},
                                {"label": "Undergraduate", "value": "Undergraduate"},
                                {"label": "Postgraduate", "value": "Postgraduate"},
                            ],
                            value="Undergraduate"
                        ),
                    ], md=4),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Number of Children"),
                        dbc.Input(type="number", id="children", placeholder="Enter Number of Children", min=0, required=True),
                    ], md=4),
                ], className="mb-3"),

                dbc.Button("Submit", id="submit_demography", color="primary", className="mt-2"),
                html.Div(id="form_output", className="mt-3 text-success"),
            ])
        ],
        body=True,
        className="mb-4 shadow-sm"
    )


# Load the trained model
model = joblib.load("student_performance_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")


def PerformanceImpactChart(df: pd.DataFrame, label_encoders: dict):
    # Encode categorical columns
    categorical_cols = [
        "StudentID","Marital Status","Employment Status","Gender","Socioeconomic Status",
        "Location","District","Education Level","Javascript","Python","HCD","Communication",
        "Course Completion","Activity","Participation Status","Role","Start Date","End Date","Date"
    ]
    df_enc = df.copy()
    for col in categorical_cols:
        if col in df_enc.columns and col in label_encoders:
            df_enc[col] = label_encoders[col].transform(df_enc[col].astype(str))

    # Select features used during training
    feature_cols = model.feature_names_in_
    X = df_enc[feature_cols].fillna(0)

    # SHAP explainer (without check_additivity)
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)

    # Aggregate impact
    import numpy as np
    import pandas as pd
    import plotly.express as px

    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"Feature": feature_cols, "Impact": importance}).sort_values("Impact", ascending=False)

    # Plotly figure
    fig = px.bar(
        importance_df,
        x="Impact",
        y="Feature",
        orientation="h",
        title="Impact of Features on Student Performance",
        color="Impact",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))

    return fig


def WhatIfPerformanceComponent(df: pd.DataFrame, label_encoders: dict, height: int = 600):
    """
    Returns a Dash dbc.Col containing sliders for what-if analysis and a SHAP bar chart.
    """
    # Define a unique ID for callbacks
    component_id = "whatif-performance"

    layout = dbc.Col([
        html.H5("Student Performance What-If Analysis"),

        # Attendance % slider
        html.Label("Attendance %"),
        dcc.Slider(
            id=f"{component_id}-attendance-slider",
            min=0,
            max=100,
            step=1,
            value=df["Attendance %"].mean(),
            marks={i: str(i) for i in range(0, 101, 10)}
        ),

        # Hours per week slider
        html.Label("Hours Per Week"),
        dcc.Slider(
            id=f"{component_id}-hours-slider",
            min=0,
            max=20,
            step=0.5,
            value=df["Hours Per Week"].mean(),
            marks={i: str(i) for i in range(0, 21, 2)}
        ),

        # SHAP graph
        dcc.Graph(
            id=f"{component_id}-shap-graph",
            style={"height": f"{height}px"}
        )
    ], width=6)

    # Create the callback for interactivity
    @app.callback(
        Output(f"{component_id}-shap-graph", "figure"),
        Input(f"{component_id}-attendance-slider", "value"),
        Input(f"{component_id}-hours-slider", "value")
    )
    def update_shap(attendance_val, hours_val):
        # Copy the dataframe
        df_copy = df.copy()
        # Apply sliders to first student for simplicity
        df_copy.loc[0, "Attendance %"] = attendance_val
        df_copy.loc[0, "Hours Per Week"] = hours_val

        # Encode categorical columns
        categorical_cols = [
            "StudentID","Marital Status","Employment Status","Gender","Socioeconomic Status",
            "Location","District","Education Level","Javascript","Python","HCD","Communication",
            "Course Completion","Activity","Participation Status","Role","Start Date","End Date","Date"
        ]
        df_enc = df_copy.copy()
        for col in categorical_cols:
            if col in df_enc.columns and col in label_encoders:
                df_enc[col] = label_encoders[col].transform(df_enc[col].astype(str))

        # Select features used in training
        feature_cols = model.feature_names_in_
        X = df_enc[feature_cols].fillna(0)

        # SHAP explainer
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X)

        # Aggregate impact
        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({"Feature": feature_cols, "Impact": importance}).sort_values("Impact", ascending=False)

        # Plotly figure
        fig = px.bar(
            importance_df,
            x="Impact",
            y="Feature",
            orientation="h",
            title=f"Predicted Performance Score: {model.predict(X)[0]:.2f}",
            color="Impact",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))

        return fig

    return layout

# Configure Gemini API
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("API"))

def GeminiQnA(df: pd.DataFrame, component_id: str = "gemini-qna"):
    """
    Creates a reusable Gemini Q&A component for a dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame to query
        component_id (str): base id for Dash components (to allow multiple instances)
    
    Returns:
        html.Div: Dash layout component
    """
    return html.Div([
        html.H3("Ask Questions About the Data"),
        
        dcc.Textarea(
            id=f"{component_id}-input",
            placeholder="Ask a question about the dataset...",
            style={"width": "100%", "height": "100px"}
        ),
        html.Button("ASK AI", id=f"{component_id}-btn", n_clicks=0),
        
        html.Div(id=f"{component_id}-output", 
                 style={"marginTop": "20px", "whiteSpace": "pre-wrap"})
    ])


def register_callbacks(app, df: pd.DataFrame, component_id: str = "gemini-qna"):
    """
    Registers the callbacks for a Gemini Q&A component instance.
    """
    @app.callback(
        Output(f"{component_id}-output", "children"),
        Input(f"{component_id}-btn", "n_clicks"),
        State(f"{component_id}-input", "value"),
        prevent_initial_call=True
    )
    def ask_gemini(n, question):
        if not question:
            return "Please enter a question."
        
        try:
            # Build prompt
            prompt = f"""
            You are a data assistant. 
            Answer the question using the pandas dataframe provided below.

            DataFrame (sample):
            {df.head(10).to_string()}

            Question: {question}

            Provide a clear answer.
            """

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)

            return response.text
        
        except Exception as e:
            return f"Error: {str(e)}"

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.Div([
        html.Img(src='assets/refactory_logo.png', style={'height': '50px'}),
        html.H4("Refactory Student Analysis Dashboard", className="my-3"),
    ], className="d-flex align-items-center gap-3"),

      #  CARDS
    create_kpi_cards(merged_df),
    dbc.Row([
        dbc.Col(GradePieChart(merged_df), width=6),
        dbc.Col(GradeBoxplot(melted), width=6),
    ]),

    dcc.Graph(figure=fig_std, className="chart-card"),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_missing, style={"height": "600px"}, className="chart-card")),
        dbc.Col(dcc.Graph(figure=fig_violin, style={"height": "600px"},  className="chart-card")),
    ]),

    html.H4("Behavioral", className="my-3"),
    create_engagement_cards(merged_df),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_time, style={"height": "600px"}, className="chart-card")),
        dbc.Col(dcc.Graph(figure=fig_msgs, style={"height": "600px"}, className="chart-card")),
    ]),
    dcc.Graph(figure=fig_forum, className="chart-card"),
    dcc.Graph(figure=fig_assign, className="chart-card"),

    html.H4("Demography", className="my-3"),
    dcc.Graph(figure=fig_gender, className="chart-card"),
    dcc.Graph(figure=fig_income, className="chart-card"),
    dcc.Graph(figure=fig_employment, className="chart-card"),
    dcc.Graph(figure=fig_district, className="chart-card"),
    dcc.Graph(figure=fig_edu, className="chart-card"), 
    dcc.Graph(figure=fig_location_study, className="chart-card"), 


    html.H4("Extracurricular Activity", className="my-3"),
    dcc.Graph(figure=fig_leadership, className="chart-card"), 
    dcc.Graph(figure=fig_radar, className="chart-card"), 
    dcc.Graph(figure=fig_sunburst, className="chart-card"), 
    dcc.Graph(figure=fig_heat, className="chart-card"),
    # dcc.Graph(figure=fig_archetype_pie),
    dcc.Graph(figure=fig_success),


    html.H4("Course Design Insights", className="my-3"),
    dcc.Graph(figure=fig_na, className="chart-card"),
    dcc.Graph(figure=fig_corr, className="chart-card"),
    dcc.Graph(figure=fig_util, className="chart-card"),
    dcc.Graph(figure=fig_support, className="chart-card"),
    dcc.Graph(figure=fig_access, className="chart-card"),
    dcc.Graph(figure=fig_forum_2, className="chart-card"),

    html.H4("Temporal Trend Analysis", className="my-3"),
    dcc.Graph(figure=fig_forum_2, className="chart-card"),

    dbc.Col(DemographyForm(), width=12),

    # Add Gemini component to app layout
    dbc.Col(GeminiQnA(merged_df, "student-qna"), width=12),

# Register its callbacks
    dbc.Col(register_callbacks(app, merged_df, "student-qna")),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                figure=PerformanceImpactChart(merged_df, label_encoders),
                className="chart-card",
                style={"height": "600px"}
            ), width=6
        ),

    ]),
    dbc.Col(WhatIfPerformanceComponent(merged_df, label_encoders))

])
if __name__ == "__main__":
    app.run(debug=True)





