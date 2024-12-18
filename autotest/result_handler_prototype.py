import os
import pandas as pd
import pyemu 
# class ResultFile(object):
# 	def __init__(self,resulttype):
# 		if resulttype == "par_ensembles":
# 			print(resulttype)
# 		#print(stuff)


class ResultHandler(object):
    def __init__(self,m_d,case):
        self.m_d = m_d
        self.case = case
        self.path2files = [os.path.join(self.m_d,f.lower()) for f in os.listdir(self.m_d)]
        self.results_loaded = {}
        #todo: check that m_ds are infact valid master dirs

    def get_ensemble_files(self,tag):
        files = []
        case_tag = self.case + tag

        for f in self.path2files:
            if tag in f.lower() and\
                os.path.split(f)[1].lower().startswith(self.case+".") and \
                len(os.path.split(f)[1].split(".")) == 4:
                try:
                    itr = int(os.path.split(f)[1].split(".")[1])
                except Exception as e:
                    pass
                else:
                    files.append(f)
            elif case_tag in os.path.split(f)[1].lower():
                files.append(f)
        return files

    def get_or_load_csv_file(self,filename,index_col=None):
        if len(os.path.split(filename)[0]) == 0:
            filename = os.path.join(self.m_d,filename)
        if filename in self.results_loaded:
            return self.results_loaded[filename]
        else:
            df = pd.read_csv(filename,index_col=index_col)
            self.results_loaded[filename] = df
            return df

    def get_or_load_ensemble_file(self, filename):
        if len(os.path.split(filename)) == 1:
            filename = os.path.join(self.m_d,filename)
        if filename in self.results_loaded:
            return self.results_loaded[filename]
        else:
            if filename.lower().endswith(".csv"):
                df = pd.read_csv(filename, index_col=0)
            elif filename.lower().endswith(".jcb"):
                df = pyemu.Matrix.from_binary(filename).to_dataframe()
            self.results_loaded[filename] = df

            return df



class ResultIesHandler(ResultHandler):

    def __getattr__(self,tag):
        tag = tag.lower().strip()
        print(tag)
        if tag.startswith("par_en") or tag.startswith("obs_en"):
            itr = self.get_ensemble_iter(tag)
            ttag = tag.split("_")[0]
            #load the combined ensemble
            if itr is None:
                file_tag = ".{0}.".format(ttag)
                files = self.get_ensemble_files(file_tag)
                print(files)
                if len(files) == 0:
                    raise Exception()
                itrs = [int(os.path.split(f)[1].split('.')[1]) for f in files]
                d = {i:f for i,f in zip(itrs,files)}
                itrs.sort()
                dfs = []
                for itr in itrs:
                    df = self.get_or_load_ensemble_file(d[itr])
                    dfs.append(df)
                df = pd.concat(dfs,keys=itrs,names=["iteration","real_name"])
                return df
            else:
                file_tag = ".{0}.{1}.".format(itr,ttag)
                files = self.get_ensemble_files(file_tag)
                if len(files) != 1:
                    #todo something here...
                    raise Exception()

                df = self.get_or_load_ensemble_file(files[0])
                return df


        elif tag.startswith("phi"):
            phi_type = tag.split("_")[1]
            csv_filename = "{0}.phi.{1}.csv".format(self.case,phi_type)
            df = self.get_or_load_csv_file(csv_filename)
            return df

        elif tag.startswith("noise_en"):
            files = self.get_ensemble_files(".obs+noise.")
            if len(files) != 1:
                raise Exception()
            df = self.get_or_load_ensemble_file(files[0])
            return df
        elif tag.startswith("weight_en"):
            files = self.get_ensemble_files(".weights.")
            if len(files) != 1:
                raise Exception()
            df = self.get_or_load_ensemble_file(files[0])
            return df

        elif tag.startswith("pdc"):
            itr = self.get_ensemble_iter(tag)
            ttag = tag.split("_")[0]
            # load the combined ensemble
            if itr is None:
                file_tag = ".{0}.".format(ttag)
                files = self.get_ensemble_files(file_tag)
                print(files)
                if len(files) == 0:
                    raise Exception()
                itrs = [int(os.path.split(f)[1].split('.')[1]) for f in files]
                d = {i: f for i, f in zip(itrs, files)}
                itrs.sort()
                dfs = []
                for itr in itrs:
                    df = self.get_or_load_csv_file(d[itr],index_col=0)
                    print(df)
                    dfs.append(df)
                df = pd.concat(dfs, keys=itrs, names=["iteration", "name"])
                return df
            else:
                file_tag = ".{0}.{1}.".format(itr, ttag)
                files = self.get_ensemble_files(file_tag)
                if len(files) != 1:
                    # todo something here...
                    raise Exception()

                df = self.get_or_load_ensemble_file(files[0])
                return df

        elif tag.startswith("pcs"):
            itr = self.get_ensemble_iter(tag)
            ttag = tag.split("_")[0]
            # load the combined ensemble
            if itr is None:
                file_tag = ".{0}.".format(ttag)
                files = self.get_ensemble_files(file_tag)
                print(files)
                if len(files) == 0:
                    raise Exception()
                itrs = [int(os.path.split(f)[1].split('.')[1]) for f in files]
                d = {i: f for i, f in zip(itrs, files)}
                itrs.sort()
                dfs = []
                for itr in itrs:
                    df = self.get_or_load_csv_file(d[itr],index_col=0)
                    print(df)
                    dfs.append(df)
                df = pd.concat(dfs, keys=itrs, names=["iteration", "group"])
                return df
            else:
                file_tag = ".{0}.{1}.".format(itr, ttag)
                files = self.get_ensemble_files(file_tag)
                if len(files) != 1:
                    # todo something here...
                    raise Exception()

                df = self.get_or_load_ensemble_file(files[0])
                return df



        else:
            raise Exception("tag: '{0}' not recognized".format(tag))


    def get_ensemble_iter(self,tag):
        if "ensemble" in tag:
            itr = tag.split("ensemble")[1]
        elif "en" in tag:
            itr = tag.split("en")[1]
        elif tag == "pdc":
            itr = tag.split("pdc")[1]
        elif tag == "pcs":
            itr = tag.split("pcs")[1]
        else:
            raise Exception()
        if itr == "":
            itr = None
        return itr

class ResultMouHandler(ResultHandler):
    def __getattr__(self,tag):
        print(tag)
        #return ResultFile(tag)
        pass


class Results(object):
    def __init__(self,m_d,case=None):
        #todo: if case is none, look for one and only one control file in m_d

        self.m_d = m_d
        self.case = case
        self.ieshand = ResultIesHandler(self.m_d,self.case)
        self.mouhand = ResultMouHandler(self.m_d,self.case)


    def __getattr__(self,tag):

        if tag == "ies":
            return self.ieshand
        elif tag == "mou":
            return self.mouhand
        else:
            raise Exception()


r = Results(m_d=os.path.join("pst","master_ies1"),case="pest")
#get all change sum files in an multiindex df
print(r.ies.pcs)
# same for conflicts across iterations
print(r.ies.pdc)
print(r.ies.weight_en)
print(r.ies.phi_lambda)
print(r.ies.phi_group)
print(r.ies.phi_actual)
print(r.ies.phi_meas)
print(r.ies.noise_en)
#get the prior par en
print(r.ies.par_en0)
# get the 1st iter obs en
print(r.ies.obs_ensemble1)
#get the combined par en across all iters
print(r.ies.par_en)