
def merge_dfs(dataframes):
    df = dataframes[0]
    for dataframe in dataframes[1:]:
        df = df.append(dataframe)
    return df