#!/usr/bin/env python
# coding: utf-8

# In[2]:


from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd


# In[3]:


# Load data
import pandas as pd 
demographic_df = pd.read_csv('data/demographics.csv')
academic_df = pd.read_csv('data/academicPerformance.csv')
activities_df = pd.read_csv('data/extracurricularActivities.csv')
behavior_df = pd.read_csv('data/behavioralPatterns.csv')


# In[4]:


demographic_df.head()


# In[5]:


academic_df


# In[6]:


activities_df


# In[7]:


behavior_df


# In[8]:


# Rename ID columns to StudentID for consistency
demographic_df.rename(columns={'ID': 'StudentID'}, inplace=True)
academic_df.rename(columns={'Student ID': 'StudentID'}, inplace=True)


# In[9]:


merged_df = demographic_df.merge(academic_df, on='StudentID') \
                          .merge(activities_df, on='StudentID') \
                          .merge(behavior_df, on='StudentID')


# In[10]:


pd.set_option('display.max_columns', None)
merged_df.head()


# In[11]:


merged_df.shape


# In[12]:


print("Duplicated rows:",merged_df.duplicated().sum())


# In[13]:


missing_representations = ['NA', 'N/A', '', 'na', 'n/a', 'NaN']
missing_check = merged_df.isin(missing_representations) | merged_df.isnull()
missing_summary = missing_check.sum().sort_values(ascending=False)
print("Missing values per column (including text forms):\n", missing_summary[missing_summary > 0])


# In[14]:


cat_columns = merged_df.select_dtypes(include="object").columns
print("\n🔤 Categorical column unique values:")
for col in cat_columns:
    print(f"- {col}: {merged_df[col].nunique()} unique values")


# In[15]:


# Student with missing grades 
merged_df['Missing grades'] = merged_df[['Python', 'HCD', 'Communication']].isnull().any(axis=1)


# In[16]:


# How many student missing grades for each course
merged_df[['Python', 'HCD', 'Communication']].isnull().sum()


# In[17]:


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

# In[18]:


grade_map = {'A':4.0, 'A-':3.7, 'B+':3.3, 'B':3.0, 'B-':2.7, 'C+':2.3, 'C':2.0, 'N/A': None}
grade_cols = ['Javascript', 'Python', 'HCD', 'Communication']
for col in grade_cols:
    merged_df[col + '_num'] = merged_df[col].map(grade_map)


# In[19]:


# Melt for some visuals
melted = merged_df.melt(id_vars=['StudentID', 'Course Completion'], 
                 value_vars=[col + '_num' for col in grade_cols], 
                 var_name='Course', value_name='Grade')
melted['Course'] = melted['Course'].str.replace('_num', '')


# In[35]:


# Grade distribution boxplot
fig_box = px.box(
    melted,
    x='Course',
    y='Grade',
    color='Course', 
    color_discrete_sequence=['#55c3c7', '#744674', '#684c64']
)

# Update the layout (styling)
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

# fig_box.show()


# In[41]:


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


# In[45]:


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


# In[47]:


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

# In[57]:


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
fig_time.show()


# In[64]:


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
fig_forum.show()


# In[71]:


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

fig_msgs.show()


# In[76]:


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

fig_assign.show()


# Performance Gap Identification

# In[79]:


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

fig_gender.show()


# In[83]:


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

fig_income.show()


# In[90]:


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

fig_employment.show()


# In[100]:


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

# fig_district.show()


# In[105]:


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

fig_kids.show()


# In[108]:


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

fig_marital.show()


# In[112]:


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

fig_edu.show()


# In[118]:


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

fig_location_study.show()


# Extracurricular Activity Insights

# In[124]:


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

fig_participation_pie.show()

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

fig_participation_perf.show()


# In[137]:


# Leadership Roles Correlation 
fig_leadership = px.bar(
    merged_df.groupby("Role")["Average Grade"].mean().reset_index(),
    x="Role", y="Average Grade",
    title="Academic Performance by Leadership Role",
    color="Average Grade",
    color_continuous_scale="teal"
)

fig_leadership.show()

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
fig_radar.show()


# In[ ]:


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

fig_violin.show()


# In[164]:


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

fig_sunburst.show()


# In[167]:


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

fig_heat.show()


# In[126]:


print(merged_df.columns.tolist())


# In[176]:


custom_color_map = {
    "Engaged High Achievers": "#2a9d8f",
    "Struggling Engagers": "#643464",
    "At-Risk Students": "#4ca4c8",
    "Natural Talents": "#b494b4",
    "Others": "#e2dce4"
}

fig_archetype_pie = px.pie(
    merged_df,
    names="Archetype",
    title="Distribution of Student Archetypes",
    hole=0.4,
    color="Archetype",  # this tells Plotly which column to map
    color_discrete_map=custom_color_map
)

# Optional: pull out at-risk students to highlight
fig_archetype_pie.update_traces(
    textinfo='percent+label',
    pull=[0.05 if val == "At-Risk Students" else 0 for val in merged_df["Archetype"]]
)

# Custom background and font
fig_archetype_pie.update_layout(
    paper_bgcolor='#f9fafb',
    title_font_color='#264653',
    font=dict(color='#264653', size=13)
)

fig_archetype_pie.show()


# In[ ]:


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

fig_success.show()


# Course Design Insights

# In[201]:


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

fig_na.show()


# In[208]:


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

fig_corr.show()


# In[209]:


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

fig_util.show()


# In[211]:


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

fig_support.show()


# In[212]:


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

fig_access.show()


# In[214]:


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

fig_forum_2.show()


# Temporal Trend Analysis
# 
# Time-Based Patterns

# In[217]:


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

fig_seasonal.show()


# In[218]:


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

fig_weekly.show()


# In[219]:


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

fig_progress.show()


# In[220]:


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

fig_dropout.show()


# In[ ]:


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.Div([
        html.Img(src='assets/refactory_logo.png', style={'height': '50px'}),
    ], className="d-flex align-items-center gap-3"),
    html.H4("Refactory Student Analysis Dashboard", className="my-3"),

    dcc.Graph(figure=fig_box),
    dcc.Graph(figure=fig_std),
    dcc.Graph(figure=fig_missing),
    dcc.Graph(figure=fig_violin),

    html.H4("Behavioral", className="my-3"),
    dcc.Graph(figure=fig_time),
    dcc.Graph(figure=fig_forum),
    dcc.Graph(figure=fig_msgs),
    dcc.Graph(figure=fig_assign),

    html.H4("Demography", className="my-3"),
    dcc.Graph(figure=fig_gender),
    dcc.Graph(figure=fig_income),
    dcc.Graph(figure=fig_employment),
    dcc.Graph(figure=fig_district),
    dcc.Graph(figure=fig_edu), 
    dcc.Graph(figure=fig_location_study), 


    html.H4("Extracurricular Activity", className="my-3"),
    dcc.Graph(figure=fig_leadership), 
    dcc.Graph(figure=fig_radar), 
    dcc.Graph(figure=fig_sunburst), 
    dcc.Graph(figure=fig_heat),
    dcc.Graph(figure=fig_archetype_pie),
    dcc.Graph(figure=fig_success),


    html.H4("Course Design Insights", className="my-3"),
    dcc.Graph(figure=fig_na),
    dcc.Graph(figure=fig_corr),
    dcc.Graph(figure=fig_util),
    dcc.Graph(figure=fig_support),
    dcc.Graph(figure=fig_access),
    dcc.Graph(figure=fig_forum_2),

    html.H4("Temporal Trend Analysis", className="my-3"),
    dcc.Graph(figure=fig_forum_2),


])

app.run(debug=True)


# In[ ]:





# In[ ]:





# In[ ]:




