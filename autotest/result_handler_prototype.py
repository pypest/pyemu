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
        self.failed2load_files = []
        #todo: check that m_ds are infact valid master dirs



    def get_or_load_csv_file(self,filename,index_col=None):
        if len(os.path.split(filename)[0]) == 0:
            filename = os.path.join(self.m_d,filename)
        if filename in self.results_loaded:
            return self.results_loaded[filename]
        else:
            df = None
            try:
                df = pd.read_csv(filename,index_col=index_col)
                self.results_loaded[filename] = df
            except Exception as e:
                print("error loading file '{0}': {1}".format(filename, str(e)))
                self.failed2load_files.append([filename, str(e)])
            return df

    def get_or_load_ensemble_file(self, filename,index_name="realization"):
        if len(os.path.split(filename)) == 1:
            filename = os.path.join(self.m_d,filename)
        if filename in self.results_loaded:
            return self.results_loaded[filename]
        else:
            df = None
            try:
                if filename.lower().endswith(".csv"):
                    df = pd.read_csv(filename, index_col=0)
                elif filename.lower().endswith(".jcb") or filename.lower().endswith(".jco") or filename.lower().endswith(".bin"):
                    df = pyemu.Matrix.from_binary(filename).to_dataframe()
                df.index.name = index_name
                self.results_loaded[filename] = df
            except Exception as e:
                print("error loading file '{0}': {1}".format(filename,str(e)))
                self.failed2load_files.append([filename,str(e)])
            return df

    def check_dup_iters(self,itrs, files):
        dup_itrs,dup_files = [],[]
        for i,itr in enumerate(itrs[:-1]):
            if itr in itrs[i+1:]:
                dup_itrs.append(str(itr))
                dup_files.append(files[i])
        if len(dup_files) > 0:
            print("dup files: {0}".format("\n".join(dup_files)))
            print("dup itrs: {0}".format("\n".join(dup_itrs)))
            raise Exception("duplicate iteration tags found")



class ResultIesHandler(ResultHandler):

    def get_files(self,tag):
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

    def parse_iter_from_tag(self,tag):
        if "ensemble" in tag:
            itr = tag.split("ensemble")[1]
        elif "en" in tag:
            itr = tag.split("en")[1]
        elif tag == "pdc":
            itr = tag.split("pdc")[1]
        elif tag == "pcs":
            itr = tag.split("pcs")[1]
        else:
            raise Exception("parse_iter_from_tag: unrecognized tag: '{0}'".format(tag))
        if itr == "":
            itr = None
        return itr

    def __getattr__(self,tag):
        tag = tag.lower().strip()
        if tag.startswith("par_en") or tag.startswith("obs_en"):
            itr = self.parse_iter_from_tag(tag)
            ttag = tag.split("_")[0]
            #load the combined ensemble
            if itr is None:
                file_tag = ".{0}.".format(ttag)
                files = self.get_files(file_tag)
                if len(files) == 0:
                    raise Exception()
                itrs = [int(os.path.split(f)[1].split('.')[1]) for f in files]
                self.check_dup_iters(itrs,files)
                d = {i:f for i,f in zip(itrs,files)}
                itrs.sort()
                dfs = []
                for itr in itrs:
                    df = self.get_or_load_ensemble_file(d[itr])
                    dfs.append(df)
                if len(dfs) == 0:
                    return None
                if len(dfs) > 1:
                    df = pd.concat(dfs,keys=itrs,names=["iteration","realization"])
                else:
                    df = dfs[0]
                return df
            else:
                file_tag = ".{0}.{1}.".format(itr,ttag)
                files = self.get_files(file_tag)
                if len(files) != 1:
                    #todo something here...
                    print(files)
                    raise Exception()

                df = self.get_or_load_ensemble_file(files[0])
                return df


        elif tag.startswith("phi"):
            phi_type = tag.split("_")[1]
            csv_filename = "{0}.phi.{1}.csv".format(self.case,phi_type)
            df = self.get_or_load_csv_file(csv_filename)
            return df

        elif tag.startswith("noise_en"):
            files = self.get_files(".obs+noise.")
            if len(files) != 1:
                raise Exception()
            df = self.get_or_load_ensemble_file(files[0])
            return df
        elif tag.startswith("weight_en"):
            files = self.get_files(".weights.")
            if len(files) != 1:
                raise Exception()
            df = self.get_or_load_ensemble_file(files[0])
            return df

        elif tag.startswith("pdc"):
            itr = self.parse_iter_from_tag(tag)
            ttag = tag.split("_")[0]
            # load the combined ensemble
            if itr is None:
                file_tag = ".{0}.".format(ttag)
                files = self.get_files(file_tag)
                if len(files) == 0:
                    raise Exception()
                itrs = [int(os.path.split(f)[1].split('.')[1]) for f in files]
                self.check_dup_iters(itrs, files)
                d = {i: f for i, f in zip(itrs, files)}
                itrs.sort()
                dfs = []
                for itr in itrs:
                    df = self.get_or_load_csv_file(d[itr],index_col=0)
                    dfs.append(df)
                if len(dfs) == 0:
                    return None
                if len(dfs) > 1:
                    df = pd.concat(dfs, keys=itrs, names=["iteration", "name"])
                else:
                    df = dfs[0]
                return df
            else:
                file_tag = ".{0}.{1}.".format(itr, ttag)
                files = self.get_files(file_tag)
                if len(files) != 1:
                    # todo something here...
                    raise Exception()

                df = self.get_or_load_ensemble_file(files[0])
                return df

        elif tag.startswith("pcs"):
            itr = self.parse_iter_from_tag(tag)
            ttag = tag.split("_")[0]
            # load the combined ensemble
            if itr is None:
                file_tag = ".{0}.".format(ttag)
                files = self.get_files(file_tag)
                if len(files) == 0:
                    raise Exception()
                itrs = [int(os.path.split(f)[1].split('.')[1]) for f in files]
                self.check_dup_iters(itrs, files)
                d = {i: f for i, f in zip(itrs, files)}
                itrs.sort()
                dfs = []
                for itr in itrs:
                    df = self.get_or_load_csv_file(d[itr],index_col=0)
                    dfs.append(df)
                if len(dfs) == 0:
                    return None
                if len(dfs) > 1:
                    df = pd.concat(dfs, keys=itrs, names=["iteration", "group"])
                else:
                    df = dfs[0]
                return df
            else:
                file_tag = ".{0}.{1}.".format(itr, ttag)
                files = self.get_files(file_tag)
                if len(files) != 1:
                    # todo something here...
                    raise Exception()

                df = self.get_or_load_ensemble_file(files[0])
                return df



        else:
            raise Exception("tag: '{0}' not recognized".format(tag))




class ResultMouHandler(ResultHandler):

    def get_files(self,tag):
        files = []
        case_tag = self.case + tag

        for f in self.path2files:
            if "stack" in tag and "nested" not in tag and "nested" in f:
                continue
            if "pop" in tag and "archive" not in tag and "archive" in f:
                continue
            if "pop" in tag and "chance" not in tag and "chance" in f:
                continue
            if tag in f.lower() and\
                os.path.split(f)[1].lower().startswith(self.case+"."):
                try:
                    itr = int(os.path.split(f)[1].split(".")[1])
                except Exception as e:
                    pass
                else:
                    files.append(f)
        return files

    def parse_iter_from_tag(self,tag):
        if "dvpop" in tag:
            itr = tag.split("dvpop")[1]
        elif "obspop" in tag:
            itr = tag.split("obspop")[1]
        elif "stack_summary" in tag:
            itr = tag.split("stack_summary")[1]
        elif "archive" in tag:
            itr = tag.split("archive")[1]
        elif "stack" in tag:
            itr = tag.split("stack")[1]
        else:
            raise Exception("parse_iter_from_tag: unrecognized tag: '{0}'".format(tag))
        if itr == "":
            itr = None
        return itr


    def __getattr__(self,tag):
        tag = tag.lower().strip()

        if (tag.startswith("dvpop") or tag.startswith("obspop") or \
                tag.startswith("archivedvpop") or tag.startswith("archiveobspop") \
                or tag.startswith("chancedvpop") or tag.startswith("chanceobspop")):
            itr = self.parse_iter_from_tag(tag)
            ttag = "dv_pop"
            if "obs" in tag:
                ttag = "obs_pop"
            if "archive" in tag:
                ttag = "archive."+ttag
            elif "chance" in tag:
                ttag = "chance."+ttag
            # load the combined ensemble
            if itr is None:
                file_tag = ".{0}.".format(ttag)
                files = self.get_files(file_tag)
                if len(files) == 0:
                    raise Exception("no files found for tag '{0}'".format(file_tag))
                itrs = [int(os.path.split(f)[1].split('.')[1]) for f in files]
                self.check_dup_iters(itrs, files)
                d = {i: f for i, f in zip(itrs, files)}
                itrs.sort()
                dfs = []
                for itr in itrs:
                    df = self.get_or_load_ensemble_file(d[itr],index_name="member")
                    dfs.append(df)
                if len(dfs) == 0:
                    return None
                if len(dfs) > 1:
                    df = pd.concat(dfs, keys=itrs, names=["generation", "member"])
                else:
                    df = dfs[0]
                return df
            else:
                file_tag = ".{0}.{1}.".format(itr, ttag)
                files = self.get_files(file_tag)
                if len(files) != 1:
                    # todo something here...
                    print(files)
                    raise Exception()

                df = self.get_or_load_ensemble_file(files[0],index_name="member")
                return df


        elif tag == "paretosum_archive" or tag == "paretosum":

            csv_filename = "{0}.pareto.summary.csv".format(self.case)
            if "archive" in tag:
                csv_filename = "{0}.pareto.archive.summary.csv".format(self.case)
            df = self.get_or_load_csv_file(csv_filename)
            return df


        elif tag.startswith("parstack") or tag.startswith("obsstack") or \
                tag.startswith("nestedparstack") or tag.startswith("nestedobstack"):
            itr = self.parse_iter_from_tag(tag)
            ttag = "par_stack"
            if "obs" in tag:
                ttag = "obs_stack"
            if "nested" in tag:
                ttag = "nested." + ttag
            # load the combined ensemble
            if itr is None:
                file_tag = ".{0}.".format(ttag)
                files = self.get_files(file_tag)
                if len(files) == 0:
                    raise Exception()
                itrs = [int(os.path.split(f)[1].split('.')[1]) for f in files]
                self.check_dup_iters(itrs,files)
                d = {i: f for i, f in zip(itrs, files)}
                itrs.sort()
                dfs = []
                for itr in itrs:
                    df = self.get_or_load_ensemble_file(d[itr])
                    if "nested" in tag:
                        df["realization"] = df.index.map(lambda x: x.split("||")[0])
                        df.index = df.index.map(lambda x: x.split("||")[1])

                    dfs.append(df)
                if len(dfs) == 0:
                    return None
                if "nested" in tag:
                    df = pd.concat(dfs, keys=itrs, names=["generation", "member"])
                else:
                    if len(dfs) > 1:
                        df = pd.concat(dfs, keys=itrs, names=["generation", "member"])
                    else:
                        df = dfs[0]

                return df
            else:
                file_tag = ".{0}.{1}.".format(itr, ttag)
                files = self.get_files(file_tag)
                if len(files) != 1:
                    # todo something here...
                    raise Exception()

                df = self.get_or_load_ensemble_file(files[0])
                if "nested" in tag:
                    df["realization"] = df.index.map(lambda x: x.split("||")[0])
                    df.index = df.index.map(lambda x: x.split("||")[1])
                    df.index.name = "member"
                return df


        elif tag.startswith("stack_summary"):
            itr = self.parse_iter_from_tag(tag)
            ttag = "population_stack_summary"
            # load the combined ensemble
            if itr is None:
                file_tag = ".{0}.".format(ttag)
                files = self.get_files(file_tag)
                if len(files) == 0:
                    raise Exception()
                itrs = [int(os.path.split(f)[1].split('.')[1]) for f in files]
                self.check_dup_iters(itrs, files)
                d = {i: f for i, f in zip(itrs, files)}
                itrs.sort()
                dfs = []
                for itr in itrs:
                    df = self.get_or_load_csv_file(d[itr], index_col=0)
                    dfs.append(df)
                if len(dfs) == 0:
                    return None
                if len(dfs) > 1:
                    df = pd.concat(dfs, keys=itrs, names=["generation", "member"])
                else:
                    df = dfs[0]
                return df
            else:
                file_tag = ".{0}.{1}.".format(itr, ttag)
                files = self.get_files(file_tag)
                if len(files) != 1:
                    # todo something here...
                    raise Exception()

                df = self.get_or_load_csv_file(files[0])
                return df

        else:
            raise Exception("tag: '{0}' not recognized".format(tag))


class Results(object):
    def __init__(self,m_d,case=None):
        if not os.path.exists(m_d):
            raise Exception("m_d '{0}' not found".format(m_d))
        if not os.path.isdir(m_d):
            raise Exception("m_d '{0}' is not directory".format(m_d))
        if case is None:
            pst_files = [f for f in os.listdir(m_d) if f.endswith(".pst")]
            if len(pst_files) == 0:
                raise Exception("no .pst files found in m_d '{0}'".format(m_d))
            elif len(pst_files) > 1:
                raise Exception("multiple .pst files found in m_d '{0}'".format(m_d))
            case = pst_files[0].replace(".pst","")
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



def results_ies_1_test():
    r = Results(m_d=os.path.join("pst", "master_ies1"))
    # get all change sum files in an multiindex df
    df = r.ies.pcs
    assert df is not None

    # same for conflicts across iterations
    df = r.ies.pdc
    assert df is not None

    # weights
    df = r.ies.weight_en
    #print(df)
    assert df is not None

    # various phi dfs
    df = r.ies.phi_lambda
    assert df is not None
    df = r.ies.phi_group
    assert df is not None
    df = r.ies.phi_actual
    assert df is not None
    print(df)
    df = r.ies.phi_meas
    assert df is not None
    # noise
    df = r.ies.noise_en
    assert df is not None
    # get the prior par en
    df = r.ies.par_en0
    assert df is not None
    # get the 1st iter obs en
    df = r.ies.obs_ensemble1
    assert df is not None
    # get the combined par en across all iters
    df = r.ies.par_en
    assert df is not None
    #print(df)


def results_ies_2_test():
    for case in ["test","test2"]:
        r = Results(m_d=os.path.join("pst", "master_ies2"), case=case)
        # get all change sum files in an multiindex df
        df = r.ies.pcs
        assert df is not None

        # same for conflicts across iterations
        df = r.ies.pdc
        assert df is not None

        # weights
        df = r.ies.weight_en
        assert df is not None

        # various phi dfs
        df = r.ies.phi_lambda
        assert df is not None
        df = r.ies.phi_group
        assert df is not None
        df = r.ies.phi_actual
        assert df is not None
        df = r.ies.phi_meas
        assert df is not None
        # noise
        df = r.ies.noise_en
        assert df is not None
        # get the prior par en
        df = r.ies.par_en0
        assert df is not None
        # get the 1st iter obs en
        df = r.ies.obs_ensemble1
        assert df is not None
        # get the combined par en across all iters
        df = r.ies.par_en
        assert df is not None

def results_mou_1_test():
    for m_d in [os.path.join("pst", "zdt1_bin"),os.path.join("pst", "zdt1_ascii")]:
        r = Results(m_d=m_d)

        df = r.mou.nestedparstack
        #print(df)

        assert df is not None

        df = r.mou.parstack0
        #print(df)
        assert df is not None

        df = r.mou.stack_summary0
        #print(df)
        assert df is not None


        df = r.mou.chanceobspop1
        #print(df)
        assert df is not None

        df = r.mou.chanceobspop
        #print(df)
        assert df is not None

        df = r.mou.chancedvpop1
        #print(df)
        assert df is not None

        df = r.mou.chancedvpop
        #print(df)
        assert df is not None

        df = r.mou.dvpop
        assert df is not None

        df = r.mou.dvpop0
        #print(df)
        assert df is not None

        df = r.mou.obspop
        #print(df)
        assert df is not None

        df = r.mou.obspop5
        # print(df)
        assert df is not None

        df = r.mou.paretosum_archive
        #print(df)
        assert df is not None

        df = r.mou.paretosum
        #print(df)
        assert df is not None

        df = r.mou.archivedvpop
        print(df)
        assert df is not None

        df = r.mou.archiveobspop
        #print(df)
        assert df is not None

if __name__ == "__main__":
    results_ies_1_test()
    results_ies_2_test()
    results_mou_1_test()
