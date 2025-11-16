import seaborn as sns
import matplotlib.pyplot as plt

def draw_boxplots(boxplots_per_column: int, boxplots_per_row: int, df):
    fig, axes = plt.subplots(boxplots_per_column, boxplots_per_row, figsize=(20, 25))
    axes = axes.flatten()
    features = df.loc[:, :]
    
    for i, feature in enumerate(features):
        sns.boxplot(y=df[feature], ax=axes[i])
        axes[i].set_title(feature)
    
    plt.tight_layout()
    plt.show()
    
    
def draw_hist(df):
    df.hist(bins=15, figsize=(15, 15))
    plt.suptitle("Histogram", fontsize=30, y=0.95)
    plt.show()
    

def draw_correlation_heatmap(df):
    cm = df.corr(numeric_only=True)
    plt.figure(figsize=(15, 10))
    sns.heatmap(data=cm, cmap="YlGnBu", fmt=".2f", annot=True)
    plt.title("Correlation Heatmap")
    plt.show()

def draw_pairplot(features, target): 
    sns.pairplot(
        features,
        y_vars=[target]
    )
    plt.show()
    
def handle_outliers(columns_to_handle, df):
    numeric_df = df.select_dtypes(include=['number'])
    
    q1 = numeric_df.quantile(0.25)
    q3 = numeric_df.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    
    for col in columns_to_handle:
        outliers = (df[col] < lower_bound[col]) | (df[col] > upper_bound[col])
        df[col].where(outliers != True, df[col].mean(), inplace=True)