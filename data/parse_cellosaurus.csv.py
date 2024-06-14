from Bio.ExPASy import cellosaurus
import pandas as pd
import urllib


def get_cellosaurus_Db(
    url="ftp://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt",
) -> pd.DataFrame:
    """
    Downloads Cellosaurus dataBase as pandas dataFrame
    """
    urllib.request.urlretrieve(url, "celllines.txt")
    AC, ID, AS, SY, DI, DR, BTO, CLO, SX, CA, OX, CC = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with open("celllines.txt") as handle:
        records = cellosaurus.parse(handle)
        for record in records:
            AC.append(record["AC"])  # AC
            ID.append(record["ID"])  # ID
            AS.append(record["AS"])  # secondary accession number
            SY.append(record["SY"])  # Synonym
            DI.append(record["DI"])  # Disease
            f_level = [(",").join(x) for x in record["DR"]]  # BTO data
            final_dr = (",").join(f_level)
            DR.append(final_dr)
            btos = []
            clos = []
            for x in record["DR"]:
                if x[0] == "BTO":
                    btos.append(x[1])
                elif x[0] == "CLO":
                    clos.append(x[1])
            BTO.append((",").join(btos))
            CLO.append((",").join(clos))
            SX.append(record["SX"])  # sex
            CA.append(record["CA"])  # Category
            OX.append(record["OX"])  # Organism
            CC.append(record["CC"])  # other_data
    cellosarus_df = pd.DataFrame(
        list(zip(AC, ID, AS, SY, DI, DR, BTO, CLO, SX, CA, OX, CC))
    )
    cellosarus_df.columns = [
        "AC",
        "ID",
        "AS",
        "SY",
        "DI",
        "DR",
        "BTO",
        "CLO",
        "SX",
        "CA",
        "OX",
        "CC",
    ]
    cellosarus_df.index = cellosarus_df["AC"]
    # remove AC column
    cellosarus_df.drop(["AC"], axis=1, inplace=True)
    cellosarus_df.to_csv("~/Downloads/cellosaurus_01_2024.csv")


get_cellosaurus_Db()
