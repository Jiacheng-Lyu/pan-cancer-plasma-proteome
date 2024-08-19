from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def core_PCA(df, **kwargs):
    pca = PCA(**kwargs)
    df_pca = pca.fit_transform(df)
    return df_pca, pca.explained_variance_ratio_


def core_tSNE(df, **kwargs):
    tsne = TSNE(**kwargs)
    df_tsne = tsne.fit_transform(df)
    return df_tsne


def core_UMAP(df, **kwargs):
    reducer = umap.UMAP(**kwargs)
    embedding = reducer.fit_transform(df)
    return embedding