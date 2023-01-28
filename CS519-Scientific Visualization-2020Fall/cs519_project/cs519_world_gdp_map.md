## CS519 Midterm Project Proposal -- National GDP Visualization

### Topic
To interactively visualize the GDP of each country from 1990 to 2020 on the world map with colormap

### Key Elements
1. **Data file parser:** parse CSV file provided by World Bank to retrieve GDP data of each country, 5%
2. **GDP colormap construction:** construct a colormap to give a color of a country based on the GDP value, 20%
3. **Hover display function:** hover the mouse on a country to display its specific GDP, 20%
4. **slider choose value function:** use a slider to choose the which year's GDP to display, 20%
5. **click display function:** click a country to pop up a line chart to show its GDP trend in the past 30 years, 20%
6. **line chart plotter:** draw a line chart of a country's GDP change, 10%
7. **colored map plotter:** color countries on the world map, 5%

### Requirements
- Python3.8
- jupyter notebook
- pandas
- geopandas
- bokeh
  
### Steps
1. Download [world geometry data](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/) from natural earth website and [national GDP data](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD) from world bank. 
2. Change the variable `country_shapefile` and `gdp_file` in the notebook to downloaded file path.
3. Run `bokeh serve --show cs519_world_gdp_map.ipynb` and the figure would be shown in browser as this:
![init_state.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjpzj4oabrj20rf0pywk3.jpg)
4. Hover to display current gdp of the country
![hover_gdp.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gl59f32axwj21iq138qfl.jpg)
5. Use slider to choose which year's GDP to show.
![hover_slider.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gl59fl8apwj21ie130qfm.jpg)
![gdp](https://i.imgur.com/lB4Xoz7.gif)
6. Click on a country to show its GDP trend in the last 30 years.
   
GDP of the US:
![usa.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjpzqa30y3j20rc0pwq80.jpg)

GDP of Russia:
![russia.png](http://ww1.sinaimg.cn/large/8f5d6442ly1gjpzrhjwcnj20ra0pnwjj.jpg)

### Reference
1. Bokeh Document: https://docs.bokeh.org/en/latest/docs/user_guide.html
2. Geopandas Document: https://geopandas.org/mapping.html
3. Bokeh example github: [repo](https://github.com/bokeh/bokeh/tree/b56543346bad8cdabe99960e63133619feaafbb7/examples/models)



 


