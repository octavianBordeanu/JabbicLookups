'''Imports '''

import numpy as np
import pandas as pd
from itertools import repeat
import pickle
from os import listdir
from os.path import isfile, join
from scipy import spatial
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
from gensim.models import Word2Vec
from difflib import SequenceMatcher
import collections
import inspect
import sys

class Jabbic(object):

    def __init__(self, b_fn=None, t_fn=None, f_dir=None, m_dir=None,
                 kvi=None, anchors=None, sw=None, queries=None,
                 bd=None, td=None, bm=None, tm=None,
                 qv=None, tv=None, qri=None, bdti=None, tdti=None,
                 bitw=None, titw=None, qwti=None, twti=None,
                 bva=None, q_df=None, lm=[]):

        '''
        Parameters required when creating a Jabbic class.
        ----------
        b_fn (base filename): str
             The filename of the .csv file containing the query data.
            File extention not to used but only the filename.
        t_fn (target filename): str
            The filename of the .csv file containing the data where the search of a match given a query file is to be found.
            File extention not to used but only the filename.
        f_dir (files directory): str
             The directory where csv files reside -- e.g. '/users/data'.
        m_dir (models directory): str
             The directory where the Word2Vec models reside -- e.g. '/users/trained_models'.
        kvi ( key variable index): int
            The column index of the variable for which a match is to be found.
            Take the below dataframe examples. Assume obs13, which is the query in data1, is the observation for which we want
            to find its match in data 2. Obs13 is at index 2 in data 1 row and so its match is also at index 2 in data 2 row.
            This index is called the key variable index.

                Data 1 (base data)
                | var1 | var2 | var3 | var4 | var5 |
                ------------------------------------
                |obs11 |obs12 |obs13 |obs14 |obs15 |
                ------------------------------------
                |obs21 |obs22 |obs23 |obs24 |obs25 |
                ------------------------------------
                |obs31 |obs31 |obs33 |obs34 |obs35 |
                ------------------------------------

                Data 2 (target data)
                | var1 | var2 | var3 | var4 | var5 |
                ------------------------------------
                |obs11 |obs12 |obs13 |obs14 |obs15 |
                ------------------------------------
                |obs21 |obs22 |obs23 |obs24 |obs25 |
                ------------------------------------
                |obs31 |obs31 |obs33 |obs34 |obs35 |
                ------------------------------------
        anchors: list[int, int, ...]
            Anchors are denoted by all other observations that are on the same row as the observation at the key variable index.
            For example, given data 1, obs13 is the key variable and obs11, obs12, obs14, and obs15 are the anchors.
            In this case, the anchors parameter is given as [0, 1, 3, 4]. However, it is also possible to consider fewer anchors
            in the match searching process. For example, we may be interested in finding the match of obs13 from data 1 in data 2
            based on only obs11 and obs12, in which case the anchors parameter looks like [0, 1].
        sw (semantic weight): float
            The weight given to semantic similarity in the calculation of the local match. Two types of similarities are used
            in this calculation: semantic and relational similarities. This parameter takes values between 0 and 1. If, for example,
            semantic similarity is given a weight of 0.2, then relational similarity is automatically given a weight of 0.8. This
            means that the local match should be closer to the query observation more relationally than semantically. More about these
            similarities and how they are calculated is explained in the _sem_sim and _rel_sim functions.
        queries: list[str, str, ...]
            The list of all query strings for which a local match is to be found.
            For example, for each query in [obs13, obs23, obs33] from data 1 find a match in data 2. The match in data 2 is in the
            same column as the query in data 1. For example, for obs13 in data 1, the match can only be either obs13, or obs23, or obs33
            in data 2.

        Parameters for which values are calculated within the class
        ----------
        bd (base data): pandas.core.frame.DataFrame
            Stores the csv file where queries reside in a pandas dataframe format.
        td (target data): pandas.core.frame.DataFrame
            Stores the csv file where matches are to be looked for in a pandas dataframe format.
        bm (base model): gensim.models.word2vec.Word2Vec
            Stores the Word2Vec trained model for the base data.
        tm (target model): gensim.models.word2vec.Word2Vec
            Stores the Word2Vec trained model for the target data.
        qv (query vectors): numpy.ndarray
            Stores an array of vectors for each query and its anchor points.
            For example, [obs11_vec, obs12_vec, obs13_vec, obs14_vec, obs15_vec]
                            |          |           |         |           |
                            |          |         query       |           |
                            |_________ |______  anchors  ____|___________|

        tv (target vectors): numpy.ndarray
            Stores an array of vectors for each potential match and its anchor points.
        qri (query row index): int
            Stores the row index of the query in the dataframe.
        bdti (base data to index): numpy.ndarray
            Stores an n-dimensional array where each row represents the observations by their unique id.
            For example, data 1 would look like

                Data 1 (base data)
                | var1 | var2 | var3 | var4 | var5 |
                ------------------------------------
                |  1   |   2  |   3  |   4  |  5   |
                ------------------------------------
                |  6   |   7  |   8  |   9  |  10  |
                ------------------------------------
                |  11  |  12  |  13  |  14  |  15  |
                ------------------------------------

        tdti (target data to index): numpy.ndarray
            An n-dimensional array where each row represents the observations by their unique id.

        bitw (base ids to words): OrderedDict
            Stores a dictionary of each unique id and its corresponding word in base data (e.g. {0: obs13})
        titw (target ids to words): OrderedDict
            Stores a dictionary of each unique id and its corresponding word in target data.
        bwti (base words to ids): OrderedDict
            tores a dictionary of each word in base data and its corresponding unique id (e.g. {'obs13': 0})
        twti (target words to ids): OrderedDict
            Stores a dictionary of each word in target data and its corresponding unique id.
        bva (base vectors aligned): numpy.ndarray
            Stores an n-dimensional array of base vectors aligned to the target vectors.
        q_df (queries dataframe): pandas.core.frame.DataFrame
            Stores a pandas dataframe that only contains the rows of query observations.
        lm (local matches): list
            Stores a list of lists where each sublist contains information about queries and their matches in the form
            [query observation, match observation, row index of query in base dataframe, row index of match in target dataframe,
            anchor points of query observation, anchor points of match observation, Ratcliff/Obsershelp similarity]
        '''
        self.b_fn = b_fn
        self.t_fn = t_fn
        self.f_dir = f_dir
        self.m_dir = m_dir
        self.kvi = kvi
        self.anchors = anchors
        self.sw = sw
        self.queries = queries
        self.lm = lm

        assert (None not in [*inspect.getmembers(self)[2][1].values()]), 'Jabbic arguments are missing'
        assert (len([*inspect.getmembers(self)[2][1].values()][6]) != 0), '\033[1m' + 'anchors' + '\033[0m' + ' argument requires a list of integers; the list you provided was empty'
        assert (len([*inspect.getmembers(self)[2][1].values()][8]) != 0), '\033[1m' + 'queries' + '\033[0m' + ' argument requires a list of strings (query words); the list you provided was empty'
        assert (isinstance(self.b_fn, str)), '\033[1m' + 'filename' + '\033[0m' + ' argument for the base data must be in string format and contain the .csv extention as well'
        assert (isinstance(self.t_fn, str)), '\033[1m' + 'filename' + '\033[0m' + '  argument for the target data must be in string format and contain the .csv extention as well'
        assert (isinstance(self.f_dir, str)), '\033[1m' + 'files directory' + '\033[0m' + ' argument must be in string format and be the same for both the base and target datasets'
        assert (isinstance(self.m_dir, str)), '\033[1m' + 'trained models directory' + '\033[0m' + '  argument must be in string format and be the same for both the base and target datasets'
        assert (isinstance(self.kvi, int)), '\033[1m' + 'key variable index (kvi)' + '\033[0m' + '  argument must be of type int'
        assert (isinstance(self.anchors, list)), '\033[1m' + 'anchors' + '\033[0m' + '  argument must be of type list'
        assert (isinstance(self.sw, float)), '\033[1m' + 'semmatic weight (sw)' '\033[0m' + + '  argument must be of type int'
        assert (isinstance(self.queries, list)), '\033[1m' + 'queries' + '\033[0m' + '  argument must be a list of query words'

        self.bd, self.bm = self._load_data(self.b_fn)
        self.td, self.tm = self._load_data(self.t_fn)

        self.bitw = collections.OrderedDict(enumerate(self.bm.wv.index2word))
        self.titw = collections.OrderedDict(enumerate(self.tm.wv.index2word))

        self.bwti = {w: i for i, w in self.bitw.items()}
        self.twti = {w: i for i, w in self.titw.items()}

        self.bdti = np.array([self.bwti[i] for i in self.bd.values.flatten()]).reshape(self.bd.values.shape)[:, [self.kvi] + self.anchors]
        self.tdti = np.array([self.twti[i] for i in self.td.values.flatten()]).reshape(self.td.values.shape)[:, [self.kvi] + self.anchors]

        self.bva = self._align_vectors()

        self.qri, self.tv, self.qv, self.q_df = self._row_vec_by_index()

    def _load_data(self, fname):
        model = self._load_model(f'{self.m_dir}', f'{fname}')
        data = (pd.read_csv(f'{self.f_dir}/{fname}.csv', sep=',')).dropna()

        return data, model


    def _align_vectors(self):
        ''' Train the orthogonal matrix R using the intersection words between the data at time t and t+1.

        The base space projected onto the target space. Given a query word from the base space, find its match into the
        target space.

        Orthogonal Procrustes is used to project to project the base space onto the target space. Orthogonal procrustes
        needs\pairs of equivalent/related observations in both spaces, hence the intersection between the two spaces is
        used to train the orthogonal matrix R. For example, assume the following words in both spaces:

        | base vectors | target vectors |
        --------------------------------
        | word1        | word1          |
        --------------------------------
        | word2        | word3          |
        --------------------------------
        | word3        | word3          |
        --------------------------------
        | word4        |                |
        --------------------------------
        | word5        |                |
        --------------------------------

        The orthogonal matrix R is trained on pairs (word1, word1), (word2, word2), (word3, word3). The base space
        and target space are of unequal size, meaning that word4 and word5 have no direct counterparts in the target
        space.

        However, the closest match for word4 and word5 in the target space can be found by projecting these two words
        onto the target space using the same trained projection matrix. The trained projection matrix can be used to
        project on the target space not only the observations that are paired in both spaces, but also those
        observations that are missing pairs in the target space.

        Given two matrices A and B of same shape (n x m), where n is number of observations and m number of dimensions,
        orthogonal Procrustes outputs an orthogonal matrix R (m x m) which best maps A to B. To project matrix A onto B:

            (1) calculate orthogonal matrix R using raw matrices A and B;
            (2) normalise matrices A and B;
            (3) transform matrix B using formula: A_projected = dot_product(A_normalised, R.T) * s, where s is the
                sum of singular values of dot_product(A.T, B), and R.T and A.T are the transpose of matrices R and A,
                respectively.

        More information about orthogonal Procrustes can be found here:

        Schönemann, Peter H. "A generalized solution of the orthogonal procrustes problem." Psychometrika 31, no. 1
        (1966): 1-10.

        The scipy.linalg.orthogonal_procrustes package from SciPy.org was used.

        Variables
        ----------
        intersection: list
            Stores a list of observations that are common to both the base and target data.
        tv_i (target vectors intersection): list
            Stores a list of vectors for the target observations that are also present in the base data.
        bv_i (base vectors intersection): list
            Stores a list of vectors for the base observations that are also present in the target data.
        '''
        intersection = list(self.tm.wv.vocab.keys() & self.bm.wv.vocab.keys())

        # For each space, get the vectors for the intersection words.
        tv_i = [self.tm.wv.__getitem__(i) for i in intersection]
        bv_i = [self.bm.wv.__getitem__(i) for i in intersection]

        # Train projection matrix on the intersection data.
        R, s = orthogonal_procrustes(tv_i, bv_i)

        # Align base space vectors to target space.
        bsv_a = np.dot(self.bm.wv.vectors, R.T) * s

        return bsv_a

    def _row_vec_by_index(self):
        qdbi = self.bd[self.bd.iloc[:, self.kvi].isin(self.queries)].index.values

        qvbi = list(map(self.bva.__getitem__, map(self.bdti.tolist().__getitem__, qdbi)))
        tvbi = [list(map(self.tm.wv.vectors.__getitem__, i)) for i in self.tdti]

        queries_df = self.bd[self.bd.iloc[:, self.kvi].isin(self.queries)]

        return qdbi, np.array(tvbi), np.array(qvbi), queries_df

    def _load_model(self, location, fname):
        model = Word2Vec.load(f"{location}/{fname}")
        return model

    def _save_obj(self, obj, fname):
        with open('objects/'+ fname + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def _load_obj(self, fname):
        with open('objects/' + fname + '.pkl', 'rb') as f:
            return pickle.load(f)

    def _cosine_sim(self, a, b):
        x_dot = np.dot(a, b.T)
        a_n = np.linalg.norm(a, axis = 1)
        b_n = np.linalg.norm(b, axis = 1)
        y_dot = np.dot(np.expand_dims(a_n, 1), np.expand_dims(b_n, 1).T)

        return (1 - x_dot / y_dot) / 2

    def _local_sim(self, i, j, sw):
        '''
        Local match is a combination of both semantic similarity and relational similarity.

        A lambda parameter is used to denote the weight given to each each type of similarity. The lambda parameter
        takes values between [0, 1]. The lambda parameters in this code is referred to as the semantic weight.

        The full formula for the loca match is as per below:

        local_similarity = λ*sum(semantic_similarity(qi, qi')) + (1-λ)*sum(relational_similarity(q0, qi), (q0', qi'))

            where,

            qi and qi' are vector representations of words in target data and base data, respectively;
            q0 and q0' are vector representations of words denoted by key variable index;
            qi and qi' are vector representations of words denoted by the indices given in the anchors parameter;

            q0 and qi' are aligned to the target space using orthogonal Procrustes.

        For a full description of the formulas used refer to the paper below, which is the base of this implementation.

        Zhang, Yating, Adam Jatowt, Sourav S. Bhowmick, and Katsumi Tanaka. "The past is not a foreign country:
        Detecting semantically similar terms across time." IEEE Transactions on Knowledge and Data Engineering 28, no.
        10 (2016):2793-2807.

        Parameters
        ----------
        i: int
            aligned query vector
        j: int
            target vector
        sw: float
            semantic weight given to semantic similarity
        '''
        return (sw * i + (1 - sw) * j)

    def _sem_sim(self, q_batch):
        '''
        Calculate semantic similarity.

        For example, assume the following query observations and their associated anchors:


            A: base vectors  obs11_vec: [obs12_vec, obs13_vec, obs14_vec, obs15_vec]
            B: target vectors obs21_vec: [obs22_vec, obs23_vec, obs24_vec, obs25_vec]

        The semantic similarity between (obs11_vec, obs21_vec), (obs12_vec, obs22_vec), ..., (obs15_vec, obs25_vec) is defined as

            cosine_distance(m, dot_product(q * R.T) * s),

        where m is the normalised matrix of vectors corresponding to all words in B and q is
        the normalised matrix of vectors corresponding to all words in the A.

        The semantic similarities are summed across all pairs and the minimum is taken as denoting the semantic best match in
        target data given a query word in base data.

        '''
        s_sim = []
        for i in range(6):
            a = np.array([j[i] for j in q_batch])
            b = np.array([j[i] for j in self.tv])

            s = self._cosine_sim(a, b)
            s_sim.append(s)

        s_avg = [np.mean(x, axis=0) for x in zip(*s_sim)]
        return s_avg

    def _rel_sim(self, q_batch):
        '''
        Calculate relational similarity.

        Relational similarity is the semantic similarity between two relations across time, where a relation is expressed as
        the difference between the vector of each query observation and each of its anchor points. For example,

            A: base vectors  obs11_vec: [obs12_vec, obs13_vec, obs14_vec, obs15_vec]
            B: target vectors obs21_vec: [obs22_vec, obs23_vec, obs24_vec, obs25_vec]

            relational similarity = cosine_distance((obs21 - obs22), dot((obs11 - obs12).T) * s)

        This example shows the cosine distance between the relationship between query obs11 and its anchor obs12 and between
        target observation obs21 and its anchor obs22. The same calculation is performed for

        A relationship is denoted by the difference between the query observation vector and its anchor observation vector.

        These relational similarities are summed and the minimum represents the closest relational match in target data given
        a query word in base data.

        The relational similarity is given by the formula:

                cosine_distance((m-fi), dot((q-fi').T, R) * s), where

        m and n are the words denoted by the key variable index for target data and base data, respectively;
        fi and fi' are the words denoted by the anchor point at index i for target data and base data,
        respectively.
        '''
        qv_k = np.array([i[0] for i in q_batch])
        tv_k = np.array([i[0] for i in self.tv])

        qv_c = np.array([i[1:] for i in q_batch])
        tv_c = np.array([i[1:] for i in self.tv])

        qv_sub = np.array([i - j for i, j in zip(qv_k, qv_c)])
        tv_sub = np.array([i - j for i, j in zip(tv_k, tv_c)])

        r_sim = []
        for i in range(5):
            a = np.array([j[i] for j in qv_sub])
            b = np.array([j[i] for j in tv_sub])

            s = self._cosine_sim(a, b)
            r_sim.append(s)
        r_avg = [np.mean(x, axis=0) for x in zip(*r_sim)]
        return r_avg

    def _ro_sim(self, s1, s2):
        sims = []
        for i, j in zip(np.delete(s1, [self.kvi]), np.delete(s2, [self.kvi])):
            sims.append(SequenceMatcher(None, i, j).ratio())
        return np.mean(sims)

    def find_matches(self, n_batches):
        for q_batch in np.array_split(self.qv, n_batches):
            s_sim_avg = []
            r_sim_avg = []
            s_sim_avg.extend(self._sem_sim(q_batch))
            r_sim_avg.extend(self._rel_sim(q_batch))
            l_sim = [self._local_sim(i, j, self.sw) for i, j in zip(s_sim_avg, r_sim_avg)]
            del s_sim_avg, r_sim_avg

            bm = [[self.queries[j], self.td.iloc[np.argmin(i)].values[self.kvi], j, np.argmin(i),
                   ', '.join(np.delete(self.q_df.iloc[[j]].values, self.kvi)),
                   ', '.join(np.delete(self.td.iloc[np.argmin(i)].values, self.kvi)),
                   round(self._ro_sim(self.bd.iloc[self.qri[j]].values, self.td.iloc[np.argmin(i)].values), 2)] for j, i in enumerate(l_sim)]

            self.lm.extend(bm)
