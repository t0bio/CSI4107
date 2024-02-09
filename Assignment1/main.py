from preprocess import readFiles
from index import index

def main():
    path = """<DOC>
<DOCNO> AP880212-0001 </DOCNO>
<FILEID>AP-NR-02-12-88 2344EST</FILEID>
<1ST_LINE>u i AM-Vietnam-Amnesty     02-12 0398</1ST_LINE>
<2ND_LINE>AM-Vietnam-Amnesty,0411</2ND_LINE>
<HEAD>Reports Former Saigon Officials Released from Re-education Camp</HEAD>
<DATELINE>BANGKOK, Thailand (AP) </DATELINE>
<TEXT>
   More than 150 former officers of the
overthrown South Vietnamese government have been released from a
re-education camp after 13 years of detention, the official Vietnam
News Agency reported Saturday.
   The report from Hanoi, monitored in Bangkok, did not give
specific figures, but said those freed Friday included an
ex-Cabinet minister, a deputy minister, 10 generals, 115
field-grade officers and 25 chaplains."""
    pre = readFiles(path)
    next = index(pre)
    print(next)


if __name__ == "__main__":
    main()

