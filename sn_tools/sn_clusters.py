from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

class ClusterObs:

    def __init__(self, data, nclusters, dbName, fields,RA_name='fieldRA', Dec_name='fieldDec'):
        """
        class to identify clusters of points in (RA,Dec)

        Parameters
        ----------
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
        clusters, dfclus = self.anaClusters(nclusters)

        # this is a summary of the clusters found
        self.clusters = clusters
        self.dfclusters = dfclus

        # this dataframe is a matching of initial data and clusters infos
        datadf = pd.DataFrame(np.copy(data))

        self.dataclus = datadf.merge(
            dfclus, left_on=[self.RA_name, self.Dec_name], right_on=['RA', 'Dec'])

    def makeClusters(self, nclusters):
        """
        Method to identify clusters
        It uses the KMeans algorithm from scipy

        Parameters
        ---------
        nclusters: int
         number of clusters to find

        Returns
        -------
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
        ----------
        nclusters: int
         number of clusters to consider

        Returns
        -------
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
            rclus.loc[:, 'dbName'] =self.dbName
            rclus.loc[:, 'fieldName'] = fieldName
            rclus.loc[:, 'Nvisits'] = int(Nvisits['all'])

            for key, vals in Nvisits.items():
                rclus.loc[:,'Nvisits_{}'.format(key)] = int(vals)

            rcluster = pd.concat((rcluster, rclus))

        return rcluster, dfcluster
