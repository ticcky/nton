import numpy as np
import random
import string
import os
import re
import itertools

from caminfo_tools import data_dstc2
import local_config as config


content = [
    {
        "phone": "01223 461661",
        "pricerange": "expensive",
        "addr": "31 newnham road newnham",
        "area": "west",
        "food": "indian",
        "postcode": "not available",
        "name": "india house"
    },
    {
        "addr": "cambridge retail park newmarket road fen ditton",
        "area": "east",
        "food": "italian",
        "phone": "01223 323737",
        "pricerange": "moderate",
        "postcode": "c.b 5",
        "name": "pizza hut fen ditton"
    },
    {
        "name": "michaelhouse cafe",
        "area": "centre",
        "food": "european",
        "phone": "01223 309147",
        "pricerange": "expensive",
        "postcode": "c.b 2",
        "addr": "st. michael's church trinity street city centre"
    },
    {
        "phone": "01223 506055",
        "pricerange": "expensive",
        "addr": "68 histon road chesterton",
        "area": "north",
        "food": "indian",
        "postcode": "c.b 4",
        "name": "tandoori palace"
    },
    {
        "addr": "21 burleigh street city centre",
        "pricerange": "expensive",
        "name": "hk fusion",
        "area": "centre",
        "food": "chinese",
        "phone": "not available",
        "postcode": "c.b 1"
    },
    {
        "phone": "not available",
        "pricerange": "expensive",
        "name": "saffron brasserie",
        "area": "centre",
        "food": "indian",
        "postcode": "not available",
        "addr": "not available"
    },
    {
        "phone": "01223 337766",
        "pricerange": "moderate",
        "name": "restaurant one seven",
        "area": "centre",
        "food": "british",
        "postcode": "not available",
        "addr": "de vere university arms regent street city centre"
    },
    {
        "phone": "01223 324351",
        "pricerange": "expensive",
        "name": "curry king",
        "area": "centre",
        "food": "indian",
        "postcode": "not available",
        "addr": "5 jordans yard bridge street city centre"
    },
    {
        "addr": "cambridge leisure park clifton way cherry hinton",
        "area": "south",
        "food": "italian",
        "phone": "01223 412430",
        "pricerange": "expensive",
        "postcode": "c.b 1",
        "name": "frankie and bennys"
    },
    {
        "phone": "01223 350106",
        "pricerange": "expensive",
        "name": "don pasquale pizzeria",
        "area": "centre",
        "food": "italian",
        "postcode": "not available",
        "addr": "12 market hill city centre"
    },
    {
        "phone": "01223 307581",
        "pricerange": "cheap",
        "name": "j restaurant",
        "area": "centre",
        "food": "asian oriental",
        "postcode": "c.b 2",
        "addr": "86 regent street city centre"
    },
    {
        "name": "the lucky star",
        "area": "south",
        "food": "chinese",
        "phone": "01223 244277",
        "pricerange": "cheap",
        "postcode": "c.b 1",
        "addr": "cambridge leisure park clifton way cherry hinton"
    },
    {
        "addr": "196 mill road city centre",
        "pricerange": "expensive",
        "name": "meze bar restaurant",
        "area": "centre",
        "food": "turkish",
        "phone": "not available",
        "postcode": "c.b 1"
    },
    {
        "phone": "01223 355711",
        "addr": "not available",
        "pricerange": "expensive",
        "name": "clowns cafe",
        "area": "centre",
        "food": "italian",
        "postcode": "c.b 1"
    },
    {
        "addr": "59 hills road city centre",
        "area": "centre",
        "food": "lebanese",
        "phone": "01462 432565",
        "pricerange": "moderate",
        "postcode": "c.b 2",
        "name": "ali baba"
    },
    {
        "addr": "152 - 154 hills road",
        "area": "south",
        "food": "modern european",
        "phone": "01223 413000",
        "pricerange": "moderate",
        "postcode": "c.b 2",
        "name": "restaurant alimentum"
    },
    {
        "phone": "01223 329432",
        "pricerange": "expensive",
        "name": "the golden curry",
        "area": "not available",
        "food": "indian",
        "postcode": "not available",
        "addr": "not available"
    },
    {
        "name": "cote",
        "area": "centre",
        "food": "french",
        "phone": "01223 311053",
        "pricerange": "expensive",
        "postcode": "c.b 2",
        "addr": "bridge street city centre"
    },
    {
        "name": "thanh binh",
        "area": "west",
        "food": "vietnamese",
        "phone": "01223 362456",
        "pricerange": "cheap",
        "postcode": "c.b 3",
        "addr": "17 magdalene street city centre"
    },
    {
        "name": "the nirala",
        "area": "north",
        "food": "indian",
        "phone": "01223 360966",
        "pricerange": "moderate",
        "postcode": "c.b 4",
        "addr": "7 milton road chesterton"
    },
    {
        "name": "cotto",
        "area": "centre",
        "food": "british",
        "phone": "01223 302010",
        "pricerange": "moderate",
        "postcode": "c.b 1",
        "addr": "183 east road city centre"
    },
    {
        "phone": "01223 365599",
        "pricerange": "cheap",
        "name": "zizzi cambridge",
        "area": "centre",
        "food": "italian",
        "postcode": "not available",
        "addr": "47-53 regent street"
    },
    {
        "phone": "not available",
        "pricerange": "expensive",
        "name": "panahar",
        "area": "centre",
        "food": "indian",
        "postcode": "not available",
        "addr": "not available"
    },
    {
        "phone": "01223 306306",
        "pricerange": "expensive",
        "name": "backstreet bistro",
        "area": "centre",
        "food": "gastropub",
        "postcode": "c.b 1",
        "addr": "2 sturton street city centre"
    },
    {
        "name": "yu garden",
        "area": "east",
        "food": "chinese",
        "phone": "01223 248882",
        "pricerange": "expensive",
        "postcode": "c.b 5",
        "addr": "529 newmarket road fen ditton"
    },
    {
        "phone": "01223 362433",
        "pricerange": "expensive",
        "name": "loch fyne",
        "area": "centre",
        "food": "seafood",
        "postcode": "c.b 2",
        "addr": "the little rose 37 trumpington street"
    },
    {
        "phone": "01842 753771",
        "pricerange": "cheap",
        "name": "golden house",
        "area": "centre",
        "food": "chinese",
        "postcode": "not available",
        "addr": "12 lensfield road city centre"
    },
    {
        "name": "peking restaurant",
        "area": "south",
        "food": "chinese",
        "phone": "01223 354755",
        "pricerange": "expensive",
        "postcode": "c.b 2",
        "addr": "10 homerton street city centre"
    },
    {
        "phone": "01223 323178",
        "pricerange": "expensive",
        "name": "sala thong",
        "area": "west",
        "food": "thai",
        "postcode": "c.b 3",
        "addr": "35 newnham road newnham"
    },
    {
        "name": "hakka",
        "area": "north",
        "food": "chinese",
        "phone": "01223 568988",
        "pricerange": "expensive",
        "postcode": "c.b 4",
        "addr": "milton road chesterton"
    },
    {
        "phone": "01223 352500",
        "pricerange": "expensive",
        "name": "fitzbillies restaurant",
        "area": "centre",
        "food": "british",
        "postcode": "not available",
        "addr": "51 trumpington street city centre"
    },
    {
        "name": "sitar tandoori",
        "area": "east",
        "food": "indian",
        "phone": "01223 249955",
        "pricerange": "expensive",
        "postcode": "c.b 1",
        "addr": "43 high street cherry hinton cherry hinton"
    },
    {
        "name": "eraina",
        "area": "centre",
        "food": "european",
        "phone": "01223 368786",
        "pricerange": "expensive",
        "postcode": "c.b 2",
        "addr": "free school lane city centre"
    },
    {
        "name": "taj tandoori",
        "area": "south",
        "food": "indian",
        "phone": "01223 412299",
        "pricerange": "expensive",
        "postcode": "c.b 1",
        "addr": "64 cherry hinton road cherry hinton"
    },
    {
        "name": "the good luck chinese food takeaway",
        "area": "south",
        "food": "chinese",
        "phone": "01223 244149",
        "pricerange": "expensive",
        "postcode": "c.b 1",
        "addr": "82 cherry hinton road cherry hinton"
    },
    {
        "addr": "20 milton road chesterton",
        "area": "north",
        "food": "italian",
        "phone": "01223 351707",
        "pricerange": "cheap",
        "postcode": "c.b 4",
        "name": "da vinci pizzeria"
    },
    {
        "name": "the hotpot",
        "area": "north",
        "food": "chinese",
        "phone": "01223 366552",
        "pricerange": "expensive",
        "postcode": "c.b 4",
        "addr": "66 chesterton road chesterton"
    },
    {
        "phone": "01223 566188",
        "pricerange": "moderate",
        "name": "jinling noodle bar",
        "area": "centre",
        "food": "chinese",
        "postcode": "c.b 2",
        "addr": "11 peas hill city centre"
    },
    {
        "addr": "30 bridge street city centre",
        "area": "centre",
        "food": "turkish",
        "phone": "01223 362372",
        "pricerange": "moderate",
        "postcode": "c.b 2",
        "name": "anatolia"
    },
    {
        "name": "la tasca",
        "area": "centre",
        "food": "spanish",
        "phone": "01223 464630",
        "pricerange": "moderate",
        "postcode": "c.b 2",
        "addr": "14 -16 bridge street"
    },
    {
        "phone": "01223 324033",
        "addr": "not available",
        "pricerange": "moderate",
        "postcode": "c.b 2",
        "area": "centre",
        "food": "italian",
        "name": "pizza express"
    },
    {
        "phone": "01223 361763",
        "pricerange": "cheap",
        "name": "charlie chan",
        "area": "centre",
        "food": "chinese",
        "postcode": "c.b 2",
        "addr": "regent street city centre"
    },
    {
        "phone": "01223 276182",
        "pricerange": "expensive",
        "name": "travellers rest",
        "area": "west",
        "food": "british",
        "postcode": "not available",
        "addr": "huntingdon road city centre"
    },
    {
        "name": "gourmet burger kitchen",
        "area": "centre",
        "food": "north american",
        "phone": "01223 312598",
        "pricerange": "expensive",
        "postcode": "c.b 2",
        "addr": "regent street city centre"
    },
    {
        "name": "bangkok city",
        "area": "centre",
        "food": "thai",
        "phone": "01223 354382",
        "pricerange": "expensive",
        "postcode": "c.b 2",
        "addr": "24 green street city centre"
    },
    {
        "name": "curry prince",
        "area": "east",
        "food": "indian",
        "phone": "01223 566388",
        "pricerange": "moderate",
        "postcode": "c.b 5",
        "addr": "451 newmarket road fen ditton"
    },
    {
        "addr": "newmarket road fen ditton",
        "area": "east",
        "food": "indian",
        "phone": "01223 577786",
        "pricerange": "expensive",
        "postcode": "c.b 5",
        "name": "pipasha restaurant"
    },
    {
        "name": "dojo noodle bar",
        "area": "centre",
        "food": "asian oriental",
        "phone": "01223 363471",
        "pricerange": "cheap",
        "postcode": "c.b 2",
        "addr": "40210 millers yard city centre"
    },
    {
        "phone": "01223 462354",
        "pricerange": "expensive",
        "name": "wagamama",
        "area": "centre",
        "food": "japanese",
        "postcode": "c.b 2",
        "addr": "36 saint andrews street"
    },
    {
        "name": "sesame restaurant and bar",
        "area": "not available",
        "food": "chinese",
        "phone": "not available",
        "pricerange": "expensive",
        "postcode": "not available",
        "addr": "not available"
    },
    {
        "phone": "01223 364917",
        "pricerange": "cheap",
        "name": "ask",
        "area": "centre",
        "food": "italian",
        "postcode": "c.b 2",
        "addr": "12 bridge street city centre"
    },
    {
        "name": "maharajah tandoori restaurant",
        "area": "west",
        "food": "indian",
        "phone": "01223 358399",
        "pricerange": "expensive",
        "postcode": "c.b 3",
        "addr": "41518 castle street city centre"
    },
    {
        "phone": "01223 259988",
        "pricerange": "moderate",
        "name": "riverside brasserie",
        "area": "centre",
        "food": "modern european",
        "postcode": "not available",
        "addr": "doubletree by hilton cambridge granta place mill lane"
    },
    {
        "name": "efes restaurant",
        "area": "centre",
        "food": "turkish",
        "phone": "01223 500005",
        "pricerange": "moderate",
        "postcode": "c.b 1",
        "addr": "king street city centre"
    },
    {
        "name": "yippee noodle bar",
        "area": "centre",
        "food": "asian oriental",
        "phone": "01223 518111",
        "pricerange": "moderate",
        "postcode": "c.b 1",
        "addr": "40428 king street city centre"
    },
    {
        "phone": "01223 301761",
        "pricerange": "moderate",
        "name": "shanghai family restaurant",
        "area": "centre",
        "food": "chinese",
        "postcode": "not available",
        "addr": "39 burleigh street city centre"
    },
    {
        "phone": "01223 323639",
        "pricerange": "cheap",
        "name": "kohinoor",
        "area": "centre",
        "food": "indian",
        "postcode": "not available",
        "addr": "74 mill road city centre"
    },
    {
        "phone": "01223 356060",
        "pricerange": "moderate",
        "name": "the varsity restaurant",
        "area": "centre",
        "food": "international",
        "postcode": "c.b 2",
        "addr": "35 saint andrews street city centre"
    },
    {
        "name": "cambridge lodge restaurant",
        "area": "west",
        "food": "european",
        "phone": "01223 355166",
        "pricerange": "expensive",
        "postcode": "c.b 3",
        "addr": "cambridge lodge hotel 139 huntingdon road city centre"
    },
    {
        "name": "la mimosa",
        "area": "centre",
        "food": "mediterranean",
        "phone": "01223 362525",
        "pricerange": "expensive",
        "postcode": "c.b 5",
        "addr": "thompsons lane fen ditton"
    },
    {
        "name": "la margherita",
        "area": "west",
        "food": "italian",
        "phone": "01223 315232",
        "pricerange": "cheap",
        "postcode": "c.b 3",
        "addr": "15 magdalene street city centre"
    },
    {
        "name": "the missing sock",
        "area": "east",
        "food": "international",
        "phone": "01223 812660",
        "pricerange": "cheap",
        "postcode": "c.b 25",
        "addr": "finders corner newmarket road"
    },
    {
        "phone": "01223 302330",
        "pricerange": "expensive",
        "name": "curry garden",
        "area": "centre",
        "food": "indian",
        "postcode": "not available",
        "addr": "106 regent street city centre"
    },
    {
        "addr": "grafton hotel 619 newmarket road fen ditton",
        "area": "east",
        "food": "british",
        "phone": "01223 241387",
        "pricerange": "expensive",
        "postcode": "c.b 5",
        "name": "grafton hotel restaurant"
    },
    {
        "name": "the gardenia",
        "area": "centre",
        "food": "mediterranean",
        "phone": "01223 356354",
        "pricerange": "cheap",
        "postcode": "c.b 2",
        "addr": "2 rose crescent city centre"
    },
    {
        "phone": "01223 367755",
        "pricerange": "cheap",
        "name": "rice house",
        "area": "centre",
        "food": "chinese",
        "postcode": "not available",
        "addr": "88 mill road city centre"
    },
    {
        "phone": "01223 367660",
        "pricerange": "expensive",
        "name": "bedouin",
        "area": "centre",
        "food": "african",
        "postcode": "c.b 1",
        "addr": "100 mill road city centre"
    },
    {
        "phone": "01223 323737",
        "pricerange": "cheap",
        "name": "pizza hut city centre",
        "area": "centre",
        "food": "italian",
        "postcode": "not available",
        "addr": "regent street city centre"
    },
    {
        "phone": "not available",
        "pricerange": "moderate",
        "addr": "jesus lane fen ditton",
        "area": "centre",
        "food": "not available",
        "postcode": "not available",
        "name": "pizza express fen ditton"
    },
    {
        "name": "chiquito restaurant bar",
        "area": "south",
        "food": "mexican",
        "phone": "01223 400170",
        "pricerange": "expensive",
        "postcode": "c.b 1",
        "addr": "2g cambridge leisure park cherry hinton road cherry hinton"
    },
    {
        "addr": "52 mill road city centre",
        "area": "centre",
        "food": "asian oriental",
        "phone": "01223 311911",
        "pricerange": "expensive",
        "postcode": "c.b 1",
        "name": "kymmoy"
    },
    {
        "phone": "01223 307030",
        "pricerange": "cheap",
        "name": "the river bar steakhouse and grill",
        "area": "centre",
        "food": "modern european",
        "postcode": "c.b 5",
        "addr": "quayside off bridge street"
    },
    {
        "phone": "0871 942 9180",
        "pricerange": "moderate",
        "name": "bloomsbury restaurant",
        "area": "centre",
        "food": "international",
        "postcode": "c.b 2",
        "addr": "crowne plaza hotel 20 downing street"
    },
    {
        "name": "saigon city",
        "area": "north",
        "food": "asian oriental",
        "phone": "01223 356555",
        "pricerange": "expensive",
        "postcode": "c.b 4",
        "addr": "169 high street chesterton chesterton"
    },
    {
        "name": "prezzo",
        "area": "west",
        "food": "italian",
        "phone": "01799 521260",
        "pricerange": "moderate",
        "postcode": "c.b 3",
        "addr": "21 - 24 northampton road"
    },
    {
        "phone": "01223 359506",
        "pricerange": "expensive",
        "name": "the cambridge chop house",
        "area": "centre",
        "food": "british",
        "postcode": "not available",
        "addr": "1 kings parade"
    },
    {
        "name": "royal spice",
        "area": "north",
        "food": "indian",
        "phone": "01733 553355",
        "pricerange": "cheap",
        "postcode": "c.b 4",
        "addr": "victoria avenue chesterton"
    },
    {
        "name": "saint johns chop house",
        "area": "west",
        "food": "british",
        "phone": "01223 353110",
        "pricerange": "moderate",
        "postcode": "c.b 3",
        "addr": "21 - 24 northampton street"
    },
    {
        "addr": "7 barnwell road fen ditton",
        "area": "east",
        "food": "indian",
        "phone": "01223 244955",
        "pricerange": "moderate",
        "postcode": "c.b 5",
        "name": "rajmahal"
    },
    {
        "name": "shiraz restaurant",
        "area": "centre",
        "food": "mediterranean",
        "phone": "01223 307581",
        "pricerange": "expensive",
        "postcode": "c.b 2",
        "addr": "84 regent street city centre"
    },
    {
        "name": "darrys cookhouse and wine shop",
        "area": "centre",
        "food": "modern european",
        "phone": "01223 505015",
        "pricerange": "expensive",
        "postcode": "c.b 1",
        "addr": "40270 king street city centre"
    },
    {
        "phone": "01223 352607",
        "pricerange": "expensive",
        "name": "stazione restaurant and coffee bar",
        "area": "centre",
        "food": "italian",
        "postcode": "not available",
        "addr": "market hill city centre"
    },
    {
        "phone": "01223 353942",
        "pricerange": "cheap",
        "name": "the gandhi",
        "area": "centre",
        "food": "indian",
        "postcode": "not available",
        "addr": "72 regent street city centre"
    },
    {
        "name": "little seoul",
        "area": "centre",
        "food": "korean",
        "phone": "01223 308681",
        "pricerange": "expensive",
        "postcode": "c.b 2",
        "addr": "108 regent street city centre"
    },
    {
        "name": "pizza hut cherry hinton",
        "area": "south",
        "food": "italian",
        "phone": "01223 323737",
        "pricerange": "moderate",
        "postcode": "c.b 1",
        "addr": "g4 cambridge leisure park clifton way cherry hinton"
    },
    {
        "name": "hotel du vin and bistro",
        "area": "centre",
        "food": "european",
        "phone": "01223 227330",
        "pricerange": "moderate",
        "postcode": "c.b 2",
        "addr": "15 - 19 trumpington street"
    },
    {
        "name": "restaurant two two",
        "area": "north",
        "food": "french",
        "phone": "01223 351880",
        "pricerange": "expensive",
        "postcode": "c.b 4",
        "addr": "22 chesterton road chesterton"
    },
    {
        "addr": "cambridge leisure park clifton way",
        "area": "south",
        "food": "portuguese",
        "phone": "01223 327908",
        "pricerange": "cheap",
        "postcode": "c.b 1",
        "name": "nandos"
    },
    {
        "name": "the copper kettle",
        "area": "centre",
        "food": "british",
        "phone": "01223 365068",
        "pricerange": "moderate",
        "postcode": "c.b 2",
        "addr": "4 kings parade city centre"
    },
    {
        "phone": "01223 350420",
        "pricerange": "moderate",
        "name": "lan hong house",
        "area": "centre",
        "food": "chinese",
        "postcode": "not available",
        "addr": "12 norfolk street city centre"
    },
    {
        "phone": "01223 277977",
        "pricerange": "expensive",
        "name": "graffiti",
        "area": "west",
        "food": "british",
        "postcode": "c.b 3",
        "addr": "hotel felix whitehouse lane huntingdon road"
    },
    {
        "phone": "01223 302800",
        "pricerange": "expensive",
        "name": "rice boat",
        "area": "west",
        "food": "indian",
        "postcode": "not available",
        "addr": "37 newnham road newnham"
    },
    {
        "phone": "01223 448620",
        "pricerange": "expensive",
        "name": "caffe uno",
        "area": "centre",
        "food": "italian",
        "postcode": "not available",
        "addr": "32 bridge street city centre"
    },
    {
        "phone": "01223 323361",
        "pricerange": "moderate",
        "name": "the oak bistro",
        "area": "centre",
        "food": "british",
        "postcode": "not available",
        "addr": "6 lensfield road"
    },
    {
        "phone": "01223 369299",
        "pricerange": "expensive",
        "addr": "midsummer common",
        "area": "centre",
        "food": "british",
        "postcode": "not available",
        "name": "midsummer house restaurant"
    },
    {
        "phone": "01223 356666",
        "pricerange": "moderate",
        "name": "de luca cucina and bar",
        "area": "centre",
        "food": "modern european",
        "postcode": "c.b 2",
        "addr": "83 regent street"
    },
    {
        "addr": "191 histon road chesterton",
        "area": "north",
        "food": "chinese",
        "phone": "01223 350688",
        "pricerange": "moderate",
        "postcode": "c.b 4",
        "name": "golden wok"
    },
    {
        "phone": "dontcare",
        "pricerange": "expensive",
        "name": "ugly duckling",
        "area": "centre",
        "food": "chinese",
        "postcode": "not available",
        "addr": "12 st. johns street city centre"
    },
    {
        "phone": "not available",
        "pricerange": "expensive",
        "name": "curry queen",
        "area": "not available",
        "food": "indian",
        "postcode": "c.b 1",
        "addr": "106 mill road city centre"
    },
    {
        "name": "galleria",
        "area": "centre",
        "food": "european",
        "phone": "01223 362054",
        "pricerange": "moderate",
        "postcode": "c.b 2",
        "addr": "33 bridge street"
    },
    {
        "name": "the slug and lettuce",
        "area": "centre",
        "food": "gastropub",
        "phone": "dontcare",
        "pricerange": "expensive",
        "postcode": "c.b 2",
        "addr": "34 - 35 green street"
    },
    {
        "phone": "01223 363270",
        "pricerange": "expensive",
        "name": "city stop restaurant",
        "area": "north",
        "food": "not available",
        "postcode": "not available",
        "addr": "cambridge city football club milton road chesterton"
    },
    {
        "phone": "01223 464550",
        "pricerange": "cheap",
        "addr": "4 - 6 rose crescent",
        "area": "centre",
        "food": "spanish",
        "postcode": "c.b 2",
        "name": "la raza"
    },
    {
        "name": "the cow pizza kitchen and bar",
        "area": "centre",
        "food": "gastropub",
        "phone": "01223 308871",
        "pricerange": "moderate",
        "postcode": "c.b 2",
        "addr": "corn exchange street"
    },
    {
        "name": "cocum",
        "area": "west",
        "food": "indian",
        "phone": "01223 366668",
        "pricerange": "expensive",
        "postcode": "c.b 3",
        "addr": "71 castle street city centre"
    },
    {
        "phone": "01223 327908",
        "pricerange": "cheap",
        "name": "nandos city centre",
        "area": "centre",
        "food": "portuguese",
        "postcode": "c.b 2",
        "addr": "33-34 saint andrews street"
    },
    {
        "phone": "01223 360409",
        "pricerange": "cheap",
        "name": "mahal of cambridge",
        "area": "centre",
        "food": "indian",
        "postcode": "not available",
        "addr": "3 - 5 millers yard mill lane"
    },
    {
        "name": "royal standard",
        "area": "east",
        "food": "gastropub",
        "phone": "01223 247877",
        "pricerange": "expensive",
        "postcode": "c.b 1",
        "addr": "290 mill road city centre"
    },
    {
        "addr": "205 victoria road chesterton",
        "area": "north",
        "food": "indian",
        "phone": "01223 727410",
        "pricerange": "moderate",
        "postcode": "c.b 4",
        "name": "meghna"
    }
]

content_small = [
    {
        "phone": "01223 461661",
        "pricerange": "expensive",
        "addr": "31 newnham road newnham",
        "area": "west",
        "food": "indian",
        "postcode": "not available",
        "name": "india house"
    },
    {
        "addr": "cambridge retail park newmarket road fen ditton",
        "area": "east",
        "food": "italian",
        "phone": "01223 323737",
        "pricerange": "moderate",
        "postcode": "c.b 5",
        "name": "italia house"
    },
    {
        "name": "michaelhouse cafe",
        "area": "centre",
        "food": "european",
        "phone": "01223 309147",
        "pricerange": "cheap",
        "postcode": "c.b 2",
        "addr": "st. michael's church trinity street city centre"
    },
    {
        "phone": "01223 461661",
        "pricerange": "cheap",
        "addr": "31 newnham road newnham",
        "area": "west",
        "food": "indian",
        "postcode": "not available",
        "name": "india2 house"
    },
    {
        "addr": "cambridge retail park newmarket road fen ditton",
        "area": "north",
        "food": "italian",
        "phone": "01223 323737",
        "pricerange": "moderate",
        "postcode": "c.b 5",
        "name": "italia2 house"
    },
    {
        "name": "michaelhouse2 cafe",
        "area": "east",
        "food": "european",
        "phone": "01223 309147",
        "pricerange": "cheap",
        "postcode": "c.b 2",
        "addr": "st. michael's church trinity street city centre"
    }
]


train_vocab = [u'#african', u'#ali_baba', u'#anatolia', u'#backstreet_bistro', u'#bangkok_city', u'#bedouin', u'#bloomsbury_restaurant', u'#british', u'#caffe_uno', u'#cambridge_lodge_restaurant', u'#centre', u'#cheap', u'#chinese', u'#chiquito_restaurant_bar', u'#cote', u'#curry_prince', u'#da_vinci_pizzeria', u'#dojo_noodle_bar', u'#east', u'#eraina', u'#european', u'#expensive', u'#french', u'#gastropub', u'#golden_wok', u'#grafton_hotel_restaurant', u'#hakka', u'#hotel_du_vin_and_bistro', u'#indian', u'#international', u'#italian', u'#japanese', u'#jinling_noodle_bar', u'#kohinoor', u'#korean', u'#la_margherita', u'#la_raza', u'#la_tasca', u'#lebanese', u'#little_seoul', u'#loch_fyne', u'#mahal_of_cambridge', u'#maharajah_tandoori_restaurant', u'#mediterranean', u'#meghna', u'#mexican', u'#meze_bar_restaurant', u'#michaelhouse_cafe', u'#moderate', u'#moderately', u'#nandos', u'#nandos_city_centre', u'#north', u'#peking_restaurant', u'#pipasha_restaurant', u'#pizza_hut_cherry_hinton', u'#pizza_hut_city_centre', u'#pizza_hut_fen_ditton', u'#portuguese', u'#prezzo', u'#rajmahal', u'#restaurant_alimentum', u'#restaurant_two_two', u'#royal_spice', u'#saint_johns_chop_house', u'#sala_thong', u'#seafood', u'#shiraz_restaurant', u'#south', u'#spanish', u'#thai', u'#thanh_binh', u'#the_cow_pizza_kitchen_and_bar', u'#the_gandhi', u'#the_gardenia', u'#the_lucky_star', u'#the_missing_sock', u'#the_nirala', u'#the_river_bar_steakhouse_and_grill', u'#the_varsity_restaurant', u'#turkish', u'#vietnamese', u'#wagamama', u'#west', u'#yu_garden', u',', u'.', u'7:30', u'?', u'a', u'about', u'address', u'af', u'afghan', u'afternoon', u'ah', u'air', u'airitran', u'am', u'an', u'and', u'any', u'anything', u'are', u'area', u'art', u'arts', u'asian', u'at', u'australasian', u'australian', u'austrian', u'b#ask', u'b#askaye', u'barbecue', u'barbeque', u'basque', u'bat', u'be', u'belgian', u'belgium', u'bistro', u'brazilian', u'breath', u'breathing', u'but', u'bye', u'caius', u'cambridge', u'can', u"can't", u'canape', u'canapes', u'cancun', u'cannabis', u'canope', u'cant', u'cantonese', u'care', u'caribbean', u'catalan', u'center', u'central', u'christmas', u'city', u'code', u'college', u'confirm', u'corsica', u'cough', u'could', u'creative', u'cross', u'crossover', u'cuban', u'danish', u'darling', u'did', u'do', u'dont', u'eat', u'endonesian', u'english', u'eritrean', u'euro', u'europ', u'f', u'fancy', u'ffood', u'find', u'food', u'for', u'fusion', u'gastro', u'german', u'get', u'gonville', u'good', u'goodbye', u'great', u'greek', u'halal', u'halo', u'has', u'have', u'hear', u'hello', u'hi', u'house', u'how', u'hungarian', u'i', u'iam', u'id', u'if', u'im', u'in', u'inaudible', u'indonesian', u'irish', u'is', u'it', u'jamaican', u'just', u'kind', u'ko', u'kosher', u'let', u'lets', u'like', u'looking', u'malaysian', u'matches', u'may', u'me', u'medetanian', u'mediteranian', u'mind', u'modereate', u'modertley', u'modreately', u'moment', u'more', u'moroccan', u'much', u'music', u'need', u'needs', u'new', u'nice', u'no', u'noise', u'number', u'of', u'oh', u'ok', u'okay', u'on', u'or', u'ostro', u'over', u'p', u'pan', u'panasian', u'park', u'part', u'persian', u'phone', u'place', u'please', u'polish', u'polynesian', u'postal', u'pri', u'price', u'priced', u'prices', u'pub', u'range', u'really', u'request', u'rerestaurant', u'rest', u'restaraunt', u'restaurant', u'restaurants', u'restaurnt', u'right', u'ring', u'romanian', u'russian', u'say', u'scandinavia', u'scandinavian', u'scotch', u'scottish', u'sea', u'searching', u'see', u'sells', u'ser', u'serve', u'serves', u'serving', u'should', u'side', u'sigh', u'signaporean', u'singaporean', u'so', u'some', u'something', u'sorry', u'sorry,', u'special', u'spensive', u'static', u'steak', u'steakhouse', u'swedish', u'swiss', u'system', u't', u'tasty', u'tell', u'thank', u'that', u'thats', u'the', u'there', u'theres', u'time', u'to', u'town', u'traditional', u'tran', u'trying', u'tur', u'turk', u'turkiesh', u'tuscan', u'type', u'uh', u'um', u'un', u'unintelligible', u'unusual', u'vanessa', u'vegetarian', u'venetian', u'vietna', u'wait', u'wanna', u'want', u'we', u'welcome', u'welsh', u'what', u'whats', u'with', u'wondering', u'world', u'would', u'ya', u'ye', u'yea', u'yeah', u'yes', u'york', u'you', u'your', 'not', 'bad', 'give']



class DataCamInfo(object):
    db_content = content
    fields = ["name", "area", "food", "phone", "pricerange", "postcode", "addr"]
    query_fields = ["area", "food", "pricerange"]

    def __init__(self):
        self.vocab = set()
        for i in range(10):
            #self.vocab.add(str(i))
            self.vocab.add(self.get_tagged_value("item%.2d" % i))
        for word in train_vocab:
            self.vocab.add(word)
        self.re_vocab_map = {}
        self.content = []
        for entry in self.db_content:
            t_entry = []
            for field in self.fields:
                word = entry[field]
                word_tgt = self.get_tagged_value(word)
                self.vocab.add(word_tgt)
                self.re_vocab_map[re.escape(word)] = word_tgt
                t_entry.append(word)

            self.content.append(tuple(t_entry))

        for slot, values in data_dstc2.ontology.iteritems():
            for value in values:
                word_tgt = self.get_tagged_value(value)
                self.vocab.add(word_tgt)
                self.re_vocab_map[re.escape(value)] = word_tgt

        self.vocab = list(sorted(self.vocab))

        self.re_vocab_map[re.escape('?')] = ' ?'
        self.re_vocab_map[re.escape("i'm")] = "im"
        self.re_vocab_map[re.escape("sil")] = ""
        self.re_vocab_map[re.escape("silence")] = ""
        self.re_vocab_repl = re.compile("|".join(self.re_vocab_map.keys()))

    def get_tagged_value(self, word):
        return "#" + word.replace(' ', '_')

    def get_vocab(self):
        return self.vocab

    def get_db(self):
        return self.content

    def get_db_for(self, ins, out):
        res = []
        for entry in self.db_content:
            key = tuple(self.get_tagged_value(entry[x]) for x in ins)
            val = self.get_tagged_value(entry[out])

            #res.append((key, val, ))

            keys = []
            for inp, k in zip(ins, key):
                keys.append([k, 'dontcare'])

            for key_g in itertools.product(*keys):
                res.append((key_g, val))
                break



        return res


    def gen_data(self, test_data=False, single_pass=False):
        data_dir = config.dstc_data_path
        input_dir = os.path.join(data_dir, 'dstc2/data')
        if test_data:
            flist = os.path.join(data_dir, 'dstc2/scripts/config/dstc2_test.flist')
        else:
            flist = os.path.join(data_dir, 'dstc2/scripts/config/dstc2_train.flist')
        #flist2 = os.path.join(data_dir, 'dstc2/scripts/config/dstc2_dev.flist')
        #flist3 = os.path.join(data_dir, 'dstc2/scripts/config/dstc2_test.flist')

        dialogs = data_dstc2.get_dialog_dirs(input_dir, [flist])
        cache = {}

        while dialogs:
            dialog_dir = random.choice(dialogs)

            if single_pass:
                dialogs.remove(dialog_dir)

            if dialog_dir in cache:
                res = cache[dialog_dir]
            else:
                dialog = data_dstc2.parse_dialog_from_directory(dialog_dir)

                # system = ""
                # user_utt_lst = []
                # for turn in dialog.turns:
                #     user_utt_lst.append(turn.transcription)
                #     for da in turn.output.dialog_acts:
                #         if da.act == 'offer':
                #             system = turn.output.transcript
                #             break
                #
                #     if system:
                #         break
                #user = dialog.turns[0].transcription
                user = dialog.turns[0].input.live_asr[0].hyp
                #user = " ".join(user_utt_lst)
                system = dialog.turns[1].output.transcript

                user = self._replace_entities(user)
                system = self._replace_entities(system)

                #print user
                #print system
                #print

                res = (tuple(user.split()), tuple(system.split()))

                cache[dialog_dir] = res

            if len(res[0]) != 0 and len(res[1]) != 0:
                yield res

    def gen_data_x(self, test_data=False, single_pass=False):
        train_tpls = [
            ('i want {pricerange} {food} in the {area}', '{name} is not bad'),
            ('how about something {area} that serves {pricerange} {food} food', '{name} is not bad'),
            ('could you give me {pricerange} {area} {food} food', '{name} is not bad')
        ]

        test_tpls = [
            ('looking for {pricerange} place serving {food} food located in the {area}', '{name} is not bad'),
            ('give me something {area} serving {food} food in {pricerange} pricerange', '{name} is not bad'),
            ('{pricerange} {area} {food}', '{name} is not bad')
        ]

        if test_data:
            tpls = test_tpls
            db = self.db_content[:25]
        else:
            tpls = train_tpls
            db = self.db_content[25:]

        while True:
            user, system = random.choice(tpls)

            entry = random.choice(db)
            user = str.format(user, **entry)
            system = str.format(system, **entry)

            user = self._replace_entities(user)
            system = self._replace_entities(system)

            yield (user.split(), system.split())


    def _replace_entities(self, where):
        where = where.lower()
        text = self.re_vocab_repl.sub(lambda m: self.re_vocab_map[re.escape(m.group(0))], where)

        return text

def main():
    vocab = set()
    db = DataCamInfo()
    for x in db.gen_data(single_pass=True):
        print " ".join(x[0]), "\t",  "A:", " ".join(x[1])


        #for i in x:
        #    map(vocab.add, i.split())

    #print list(sorted(vocab))
    #print len(db.get_vocab())
    #print db.get_db()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(**vars(args))