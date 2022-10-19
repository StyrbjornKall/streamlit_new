
import pandas as pd
import numpy as np
from typing import List


# Get data for model contains functions for data preprocessing in order to train a transformer model for ecotoxicity prediction.
# These utils functions are imported and used in the main script.

class PreProcessData():

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def GetOneHotEnc(self, list_of_endpoints: List[str], list_of_effects: List[str]):
        '''
        Concatenates all one hot encodings into one numpy vector. Preferably use this as input to network.
        '''
        self.dataframe = self._GetOneHotEndpoint(list_of_endpoints=list_of_endpoints)
        self.dataframe = self._GetOneHotEffect(list_of_effects=list_of_effects)

        try:
            columns = self.dataframe.filter(like='OneHotEnc').columns.tolist()
            temp1 = np.array([el.tolist() for el in self.dataframe[columns[0]].values])
            for idx, col in enumerate(columns):
                try:
                    temp2 = np.array([el.tolist() for el in self.dataframe[columns[idx+1]].values])
                    temp1 = np.concatenate((temp1,temp2), axis=1)
                except:
                    pass
            self.dataframe['OneHotEnc_concatenated'] = temp1.tolist()
        except:
            self.dataframe['OneHotEnc_concatenated'] = np.zeros((len(self.dataframe), 1)).tolist()
            print('''Will use input 0 to network due to no Onehotencodings being present.''')

        return self.dataframe

    
    def _GetOneHotEndpoint(self, list_of_endpoints: List[str]):
        '''
        Builds one hot encoded numpy arrays for given endpoints. Groups EC10 and NOEC measurements by renaming EC10 --> NOEC.
        '''
        if 'EC10' in list_of_endpoints:
            print(f"Renamed EC10 *NOEC* in {sum(self.dataframe['COMBINED_endpoint'] == 'EC10')} positions")
            self.dataframe.loc[self.dataframe.COMBINED_endpoint == 'EC10', 'COMBINED_endpoint'] = 'NOEC'
            list_of_endpoints.remove('EC10')
            
        if len(list_of_endpoints) > 1:
            hot_enc_dict = dict(zip(list_of_endpoints, np.eye(len(list_of_endpoints), dtype=int).tolist()))
            self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)
            try:
                clas = self.dataframe.COMBINED_endpoint.apply(lambda x: self._Match(x, list_of_endpoints))
                encoded_clas = clas.apply(lambda x: np.array(hot_enc_dict[x]))
                self.dataframe['OneHotEnc_endpoint'] = encoded_clas
            except:
                raise Exception('An unexpected error occurred.')

        else:
            print('''Did not return onehotencoding for Endpoint. Why? You specified only one Endpoint or you specified NOEC and EC10 which are coded to be the same endpoint.''')

        return self.dataframe



    def _GetOneHotEffect(self, list_of_effects: List[str]):
        '''
        Builds one hot encoded numpy arrays for given effects.
        '''

        if len(list_of_effects) > 1:
            hot_enc_dict = dict(zip(list_of_effects, np.eye(len(list_of_effects), dtype=int).tolist()))
            self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)
            try:
                clas = self.dataframe.COMBINED_effect.apply(lambda x: self._Match(x, list_of_effects))
                encoded_clas = clas.apply(lambda x: np.array(hot_enc_dict[x]))
                self.dataframe['OneHotEnc_effect'] = encoded_clas
            except:
                raise Exception('An unexpected error occurred.')

        else:
            print('''Did not return onehotencoding for Effect. Why? You specified only one Effect.''')

        return self.dataframe




    ## Convenience functions
    def _Match(self, x, groups):
        try:
            clas = [y for y in groups if y in x][0]
        except:
            clas = 'other'
        return clas

    def _GetSpecies(self, x):
        try:
            x = ' '.join(x.split()[0:2])
        except:
            x = 'unspecified'
        return x