# CoronaTracker-simulteModel

#### Date: Feb 2020 - March 2020
## Goal
Simulations model for outbreak prediction


### Collaborators:
- [Dominic Ligot](https://github.com/docligot)
- [Leebond](https://github.com/leebond)
- [Yiran Jing](https://github.com/YiranJing)

***
## Data Source:

### 1. Daily data (Query from DXY)
* Original Data from [Ding Xiang Yuan](https://3g.dxy.cn/newh5/view/pneumonia)。
* CSV format data from: https://github.com/BlankerL/DXY-2019-nCoV-Data CSV data file is updated frequently by [2019-nCoV Infection Data Realtime Crawler](https://github.com/BlankerL/DXY-2019-nCoV-Crawler).
* Reference from repo https://github.com/canghailan/Wuhan-2019-nCoV

#### Description
* dataset.py: Utility functions
  * Regional data (DXYArea.csv) contains all the city-level data. Data from Hong Kong SAR, Macao SAR, Tai Wan and Tibet are province-level, not city-level data.
  * Include other countries data
  * The data before 2020-02-07 are collect from other resource

#### Usage
See DXY_AreaData_query.ipynb inside `data_processing` folder, or run
```
python DXY_AreaData_query.py # output data (english version in ../data/DXYArea.csv)
```

### 2. Old (2014) dataset of flightpaths, globally

### 3. Some news, by time
