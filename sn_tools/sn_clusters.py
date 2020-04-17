from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


class ClusterObs:

    def __init__(self, data, nclusters, dbName, fields, RA_name='fieldRA', Dec_name='fieldDec'):
        """
        class to identify clusters of points in (RA,Dec)

        Parameters
        ---------------

        data: numpy record array
         data to process
        nclusters: int
         number of clusters to find
        dbName: str
         name of the file where the data where extracted from
        fields: pandas df
          fields to consider
        RA_name: str, opt
         field name for the RA (default=fieldRA)
        Dec_name: str, opt
         field name for the Dec (default=fieldDec)

        """

        # grab necessary infos
        self.data = data
        self.dbName = dbName
        self.RA_name = RA_name
        self.Dec_name = Dec_name
        self.fields = fields

        # make the cluster of points
        self.points, self.clus, self.labels = self.makeClusters(nclusters)

        # analyse the clusters
        clusters, dfclus, dataclus = self.anaClusters(nclusters)

        # this is a summary of the clusters found
        self.clusters = clusters
        self.dfclusters = dfclus
        self.dataclus = dataclus

        """
        # this dataframe is a matching of initial data and clusters infos
        datadf = pd.DataFrame(np.copy(data))

        print(datadf.columns)
        print(dfclus[['RA', 'Dec']])
        datadf[self.RA_name] = datadf[self.RA_name].round(3)
        datadf[self.Dec_name] = datadf[self.Dec_name].round(3)

        dfclus['RA'] = dfclus['RA'].round(3)
        dfclus['Dec'] = dfclus['Dec'].round(3)

        self.dataclus = datadf.merge(
            dfclus, left_on=[self.RA_name, self.Dec_name], right_on=['RA', 'Dec'])
        """

    def makeClusters(self, nclusters):
        """
        Method to identify clusters
        It uses the KMeans algorithm from scipy

        Parameters
        ---------------

        nclusters: int
         number of clusters to find

        Returns
        -----------
        points: numpy array
          array of (RA,Dec) of the points
        y_km: numpy array
          index of the clusters
        kmeans.labels_: numpy array
          kmeans label
        """

        """
        r = []
        for (pixRA, pixDec) in self.data[[self.RA_name,self.Dec_name]]:
            r.append([pixRA, pixDec])

        points = np.array(r)
        """

        points = np.array(self.data[[self.RA_name, self.Dec_name]].tolist())

        # create kmeans object
        kmeans = KMeans(n_clusters=nclusters)
        # fit kmeans object to data
        kmeans.fit(points)

        # print location of clusters learned by kmeans object
        #print('cluster centers', kmeans.cluster_centers_)

        # save new clusters for chart
        y_km = kmeans.fit_predict(points)

        return points, y_km, kmeans.labels_

    def anaClusters(self, nclusters):
        """
        Method matching clusters to data

        Parameters
        ---------------
        nclusters: int
         number of clusters to consider

        Returns
        -----------
        env: numpy record array
          summary of cluster infos:
          clusid, fieldId, RA, Dec, width_RA, width_Dec, 
          area, dbName, fieldName, Nvisits, Nvisits_all, 
          Nvisits_u, Nvisits_g, Nvisits_r, Nvisits_i, 
          Nvisits_z, Nvisits_y
        dfcluster: pandas df
          for each data point considered: RA,Dec,fieldName,clusId

        """

        rcluster = pd.DataFrame()
        dfcluster = pd.DataFrame()
        datacluster = pd.DataFrame()
        for io in range(nclusters):

            RA = self.points[self.clus == io, 0]
            Dec = self.points[self.clus == io, 1]
            dfclus = pd.DataFrame({'RA': RA, 'Dec': Dec})
            # ax.scatter(RA,Dec, s=10, c=color[io])
            indx = np.where(self.labels == io)[0]
            sel_obs = self.data[indx]
            Nvisits = getVisitsBand(sel_obs)

            min_RA = np.min(RA)
            max_RA = np.max(RA)
            min_Dec = np.min(Dec)
            max_Dec = np.max(Dec)
            mean_RA = np.mean(RA)
            mean_Dec = np.mean(Dec)
            area = np.pi*(max_RA-min_RA)*(max_Dec-min_Dec)/4.
            idx, fieldName = getName(self.fields, mean_RA)

            datacluster_loc = pd.DataFrame(np.copy(sel_obs))
            datacluster_loc.loc[:, 'fieldName'] = fieldName
            datacluster_loc.loc[:, 'clusId'] = int(io)
            datacluster_loc.loc[:, 'RA'] = mean_RA
            datacluster_loc.loc[:, 'Dec'] = mean_Dec
            datacluster = pd.concat([datacluster, datacluster_loc], sort=False)

            dfclus.loc[:, 'fieldName'] = fieldName
            dfclus.loc[:, 'clusId'] = int(io)
            dfcluster = pd.concat([dfcluster, dfclus], sort=False)

            rclus = pd.DataFrame(columns=['clusid'])
            rclus.loc[0] = int(io)
            rclus.loc[:, 'RA'] = mean_RA
            rclus.loc[:, 'Dec'] = mean_Dec
            rclus.loc[:, 'width_RA'] = max_RA-min_RA
            rclus.loc[:, 'width_Dec'] = max_Dec-min_Dec
            rclus.loc[:, 'area'] = area
            rclus.loc[:, 'dbName'] = self.dbName
            rclus.loc[:, 'fieldName'] = fieldName
            rclus.loc[:, 'Nvisits'] = int(Nvisits['all'])

            for key, vals in Nvisits.items():
                rclus.loc[:, 'Nvisits_{}'.format(key)] = int(vals)

            rcluster = pd.concat((rcluster, rclus))

        return rcluster, dfcluster, datacluster


def getVisitsBand(obs):
    """
    Function to estimate the number of visits per band
    for a set of observations

    Parameters
    ---------------
    obs: numpy record array
     array of observations

    Returns
    -----------
    Nvisits: dict
     dict with bands as keys and number of visits as values

    """

    bands = 'ugrizy'
    Nvisits = {}

    Nvisits['all'] = 0
    if 'filter' in obs.dtype.names:
        for band in bands:
            ib = obs['filter'] == band
            Nvisits[band] = len(obs[ib])
            Nvisits['all'] += len(obs[ib])
    else:
        for b in bands:
            Nvisits[b] = 0

    return Nvisits


def getName(df_fields, RA):
    """
    Function to get a field name corresponding to RA

    Parameters
    ---------------
    df_fields: pandas df
     array of fields with the following columns:
     - name: name of the field
     - fieldId: Id of the field
     - RA: RA of the field
     - Dec: Dec of the field
     - fieldnum: field number

    Returns
    ----------
    idx: int
     idx (row number) of the matching field
    name: str
     name of the matching field

    """

    _fields = df_fields.to_records(index=False)
    _idx = np.abs(_fields['RA'] - RA).argmin()

    return _idx, _fields[_idx]['name']
