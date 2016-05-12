
def fac2real_test():
    import os
    import pyemu
    pp_file = os.path.join("utils","points1.dat")
    factors_file = os.path.join("utils","factors1.dat")
    pyemu.utils.gw_utils.fac2real(pp_file,factors_file,
                                  out_file=os.path.join("utils","test.ref"))

if __name__ == "__main__":
    fac2real_test()

