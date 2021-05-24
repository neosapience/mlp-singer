import math
import re
from dataclasses import dataclass

from .constants import COD, NUC, ONS, RCD, rule_in, rule_out


def isHangul(charint):
    hangul_init = 44032
    hangul_fin = 55203
    return charint >= hangul_init and charint <= hangul_fin


def checkCharType(var_list):
    #  1: whitespace
    #  0: hangul
    # -1: non-hangul
    checked = []
    for i in range(len(var_list)):
        if var_list[i] == 32:  # whitespace
            checked.append(1)
        elif isHangul(var_list[i]):  # Hangul character
            checked.append(0)
        else:  # Non-hangul character
            checked.append(-1)
    return checked


def graph2phone(graphs):
    # Encode graphemes as utf8
    try:
        graphs = graphs.decode("utf8")
    except AttributeError:
        pass

    integers = []
    for i in range(len(graphs)):
        integers.append(ord(graphs[i]))

    # Romanization (according to Korean Spontaneous Speech corpus; 성인자유발화코퍼스)
    phones = ""

    # Pronunciation
    idx = checkCharType(integers)
    iElement = 0
    while iElement < len(integers):
        if idx[iElement] == 0:  # not space characters
            base = 44032
            df = int(integers[iElement]) - base
            iONS = int(math.floor(df / 588)) + 1
            iNUC = int(math.floor((df % 588) / 28)) + 1
            iCOD = int((df % 588) % 28) + 1

            s1 = "-" + ONS[iONS - 1]  # onset
            s2 = NUC[iNUC - 1]  # nucleus

            if COD[iCOD - 1]:  # coda
                s3 = COD[iCOD - 1]
            else:
                s3 = ""
            tmp = s1 + s2 + s3
            phones += tmp

        elif idx[iElement] == 1:  # space character
            tmp = "#"
            phones += tmp

        phones = re.sub("-(oh)", "-", phones)
        iElement += 1
        tmp = ""

    # 초성 이응 삭제
    phones = re.sub("^oh", "", phones)
    phones = re.sub("-(oh)", "", phones)

    # 받침 이응 'ng'으로 처리 (Velar nasal in coda position)
    phones = re.sub("oh-", "ng-", phones)
    phones = re.sub("oh([# ]|$)", "ng", phones)

    # Remove all characters except Hangul and syllable delimiter (hyphen; '-')
    phones = re.sub("(\W+)\-", "\\1", phones)
    phones = re.sub("\W+$", "", phones)
    phones = re.sub("^\-", "", phones)
    return phones


def phone2prono(phones):
    # Apply g2p rules
    for pattern, replacement in zip(rule_in, rule_out):
        # print pattern
        phones = re.sub(pattern, replacement, phones)
        prono = phones
    return prono


def addPhoneBoundary(phones):
    # Add a comma (,) after every second alphabets to mark phone boundaries
    ipos = 0
    newphones = ""
    while ipos + 2 <= len(phones):
        if phones[ipos] == "-":
            newphones = newphones + phones[ipos]
            ipos += 1
        elif phones[ipos] == " ":
            ipos += 1
        elif phones[ipos] == "#":
            newphones = newphones + phones[ipos]
            ipos += 1

        newphones = newphones + phones[ipos] + phones[ipos + 1] + ","
        ipos += 2

    return newphones


def graph2prono(graphs):

    romanized = graph2phone(graphs)
    romanized_bd = addPhoneBoundary(romanized)
    prono = phone2prono(romanized_bd)

    prono = re.sub(",", " ", prono)
    prono = re.sub(" $", "", prono)
    prono = re.sub("#", "-", prono)
    prono = re.sub("-+", "-", prono)

    prono_prev = prono
    identical = False
    loop_cnt = 1

    while not identical:
        prono_new = phone2prono(re.sub(" ", ",", prono_prev + ","))
        prono_new = re.sub(",", " ", prono_new)
        prono_new = re.sub(" $", "", prono_new)

        if re.sub("-", "", prono_prev) == re.sub("-", "", prono_new):
            identical = True
            prono_new = re.sub("-", "", prono_new)
        else:
            loop_cnt += 1
            prono_prev = prono_new

    return prono_new


@dataclass
class Phone:
    onset: int = None
    nucleus: int = None
    coda: int = None

    def to_list(self):
        return [idx for idx in [self.onset, self.nucleus, self.coda] if idx is not None]

    def num(self):
        return len(self.to_list())

    def __repr__(self):
        return f"({self.onset}, {self.nucleus}, {self.coda})"


def encode(graph):
    prono = graph2prono(graph)
    prono = prono.split(" ")
    encoded_prono = [Phone()]
    for p in prono:
        if p in ONS:
            encoded_prono[-1].onset = ONS.index(p) + int(ONS.index(p) < ONS.index("oh"))
        elif p in NUC:
            encoded_prono[-1].nucleus = NUC.index(p) + len(ONS)
            encoded_prono.append(Phone())
        elif p in COD:
            encoded_prono[-2].coda = RCD.index(p) + len(ONS) + len(NUC)
    return encoded_prono[:-1]


def decode(encoded_prono):
    prono = []
    for p in encoded_prono:
        phone = ""
        if p.onset is not None:
            if p.onset < ONS.index("oh"):
                phone += ONS[p.onset - 1]
            else:
                phone += ONS[p.onset]
        if p.nucleus is not None:
            phone += NUC[p.nucleus - len(ONS)]
        if p.coda is not None:
            phone += RCD[p.coda - (len(ONS) + len(NUC))]
        prono.append(phone)
    return prono
