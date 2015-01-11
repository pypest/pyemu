import os
import numpy as  np
import pylab

def schur_percent_bar(schur, ax=None, names=None):
        show = False
        if ax is None:
            fig = pylab.figure(figsize=(11,8.5))
            ax = pylab.subplot(111)
            show = True

        if schur.parcov.isdiagonal:
            prior = schur.parcov.x.flatten()
        else:
            prior = np.diag(schur.parcov.x)
        post = np.diag(schur.posterior_parameter.x)
        if names is not None:
            idx = []
            for name in names:
                i = schur.parcov.row_names.index(name.lower())
                idx.append(i)
            prior = prior[idx]
            post = post[idx]

        else:
            names = schur.parcov.row_names
        idx = np.arange(1,len(names) + 1, 1)
        reduce = 100.0 * ((prior - post) / prior)
        ax.bar(idx,reduce, width=1.0, facecolor="0.2", edgecolor="none")
        ax.legend(loc="upper left")
        ax.set_title("parameter uncertainty reduction")
        ax.set_ylabel("uncertainty reduction (%)")
        ax.set_xticks(idx + 0.5)
        ax.set_xticklabels(names, rotation=90)
        ax.set_xlim(-1, len(names) + 1)
        if show:
            pylab.show()
        return ax

def errvar_ident_bar(errvar, singular_value, ax=None, names=None):
        show = False
        if ax is None:
            fig = pylab.figure(figsize=(11,8.5))
            ax = pylab.subplot(111)
            show = True
        if names is not None:
            idx = []
            for name in names:
                i = schur.parcov.row_names.index(name.lower())
                idx.append(i)
            prior = prior[idx]
            post = post[idx]

        else:
            names = schur.parcov.row_names

def errvar_errvar_bar(errvar, singular_values, ax=None):
        raise NotImplementedError()