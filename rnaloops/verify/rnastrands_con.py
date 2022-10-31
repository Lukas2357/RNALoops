"""A module to connect RNAStrands and RNALoops databases.

    RNAStrands data must be stored in mysql database.
    Follow instructions in README of RNAStrands data to set up that mysql db.
    RNALoops data can be added to that db with below function and both can be
    queried together using below defined queries or user defined ones.

"""

import pandas as pd
from sqlalchemy import create_engine

from ..prepare.data_loader import load_data


def get_con(user='root', pwd=''):
    """Call this to connect to the db after creating it on local mysql
       Follow instructions in README of RNAStrands data to set up that mysql db

    """
    sqlEngine = create_engine(
        f"mysql+pymysql://{user}:{pwd}@localhost/sstrand_2_0", pool_recycle=3600
    )

    return sqlEngine.connect()


def populate_rnaloops(tableName="RNALoops"):
    """Call this once to create a table with RNALoops data in RNAStrands db

    """
    df = load_data('_prepared')
    con = get_con()

    try:
        df.to_sql(tableName, con, if_exists="fail")
    except ValueError as vx:
        print(vx)
    except Exception as ex:
        print(ex)
    else:
        print("Table %s created successfully." % tableName)
    finally:
        con.close()


def query_all(left_or_right='RIGHT') -> str:
    """Get all pdb ids from RNALoops joined RNAStrands

    Parameters
    ----------
    left_or_right : str, default: 'RIGHT'
        Whether to keep all RNAStrands (RIGHT) or RNALoops (LEFT) entries

    Returns
    -------
    str
        The mysql query to get all pdb ids

    """
    pdb_id = 'EXTERNAL_ID' if left_or_right == 'RIGHT' else 'home_structure'
    return f"""
        SELECT {pdb_id},
        CASE WHEN a.c1 is NULL THEN 0 ELSE a.c1 END,
        CASE WHEN b.c2 is NULL THEN 0 ELSE b.c2 END
        FROM (SELECT home_structure, count(*) c1
               FROM RNALoops
               GROUP BY home_structure) a
        {left_or_right} JOIN (
        SELECT EXTERNAL_ID, count(*) c2 FROM EXTERNAL_AND_MULTI_LOOP
        JOIN MOLECULE ON 
        MOLECULE.MOLECULE_ID = EXTERNAL_AND_MULTI_LOOP.MOLECULE_ID
        WHERE TYPE = 'multi'
        GROUP BY EXTERNAL_ID) b

        ON a.home_structure = b.EXTERNAL_ID
    """


def query_pdbid(pdb_id='2J28'):
    """Query to get RNAStrands multiloops for given pdb id

    Parameters
    ----------
    pdb_id : str, default '2J28'
        The pdb id to query

    Returns
    -------
    str
        The mysql query to get the pdb_id entry

    """
    return f"""
        SELECT * 
        FROM EXTERNAL_AND_MULTI_LOOP
        JOIN MOLECULE ON 
        MOLECULE.MOLECULE_ID = EXTERNAL_AND_MULTI_LOOP.MOLECULE_ID
        WHERE TYPE = 'multi' AND EXTERNAL_ID = '{pdb_id}'
    """


def get_matching_ids() -> pd.DataFrame:
    """Run the query to get matches of multiloop pdb ids in both databases

    Returns
    -------
    pd.DataFrame
        The df with all matching ids

    """

    con = get_con(pwd='2357')
    left = [x for x in con.execute(query_all('LEFT'))]
    right = [x for x in con.execute(query_all('RIGHT'))]
    cols = ['pdb_id', '# RNALoops Multiloops', '# RNAStrands Multiloops']
    result = pd.DataFrame(left + right, columns=cols)
    result['difference'] = (result['# RNALoops Multiloops']
                            - result['# RNAStrands Multiloops'])
    return result


def save_matching_ids(result):
    """

    Parameters
    ----------
    result : pd.DataFrame
        The result obtained from get_matching_ids

    """
    result.pdb_id = [x.lower() for x in result.pdb_id]
    result = result.drop_duplicates()
    result = result.sort_values(by=['# RNAStrands Multiloops',
                                    '# RNALoops Multiloops'], ascending=False)

    result.to_csv('RNALoops_RNAStrands_compare.csv')
