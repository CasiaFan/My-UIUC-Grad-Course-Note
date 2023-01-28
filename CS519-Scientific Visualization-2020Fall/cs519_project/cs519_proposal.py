import geopandas as gpd 
import json
import pandas as pd

# load geometry data (Download from natural earth website: https://www.naturalearthdata.com/downloads/110m-cultural-vectors/)
country_shapefile = "/Users/zongfan/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
geo_df = gpd.read_file(country_shapefile)
print(geo_df.columns)

geo_data = geo_df[["ADMIN", "ADM0_A3", "CONTINENT", "geometry"]]
print(len(geo_df))
# print(geo_df[geo_data["ADMIN"]=="France"])
geo_data.head()

# load gdp data (Download from world bank: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD)
gdp_file = "/Users/zongfan/Downloads/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_1429653/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_1429653.csv"
gdp_data = pd.read_csv(gdp_file, skiprows=[0,1,2,3])
# keep country name, code and gdp from 1980 to 2019 (2020 is NaN)
min_year = 1990
max_year = 2019
selected_years = [str(x) for  x in range(min_year, max_year+1)]
gdp_data = gdp_data[["Country Name", "Country Code"]+selected_years]
scale = 1e9  # convert unit to bilion dollars
gdp_data[selected_years] = gdp_data[selected_years]/scale
# replace NaN with 0
gdp_data = gdp_data.fillna(0)
gdp_data.head()

# merge geometry data with gdp data (use gpd merge instead of pd due to the polygon class)
# geo_gdp_df = gpd.merge(gdp_data, geo_data, how="left", left_on=["Country Name", "Country Code"], right_on=["name", "iso_a3"])
# geo_gdp_df = geo_gdp_df.drop(columns=["name", "iso_a3"])
geo_gdp_df = geo_data.merge(gdp_data, how="left", left_on=["ADM0_A3"], right_on=["Country Code"])
geo_gdp_df = geo_gdp_df.drop(columns=["ADMIN", "ADM0_A3"])
# create new columns with name year and gdp
geo_gdp_df = geo_gdp_df.melt(id_vars=["Country Name", "Country Code", "geometry", "CONTINENT"], var_name="year", value_name="gdp")
new_columns = ["country", "code", "geometry", "continent", "year", "gdp"]
geo_gdp_df.columns = new_columns
geo_gdp_df = geo_gdp_df.dropna()
geo_gdp_df.head()

from bokeh.io import output_notebook, show, output_file
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, tickers, Label, HoverTool, TapTool, PrintfTickFormatter, Slider, CustomJS
from bokeh.plotting import figure 
from bokeh.palettes import brewer, mpl, all_palettes
from bokeh.layouts import widgetbox, column

def gdp_of_year(year):
    """Keep gdp data with given year"""
    df = geo_gdp_df
    select_df = df[df["year"]==str(year)]
    return json.dumps(json.loads(select_df.to_json()))

# use turbo256 color palettes and construct color map
# print(list(all_palettes))
# print(mpl.keys())
palette = mpl["Viridis"][256][::-1] # the larger gdp, the deeper color
min_gdp=0
max_gdp=4000
# define linear color mapper
color_mapper = LinearColorMapper(low=min_gdp, high=max_gdp, palette=palette)
# define corresponding color bar
color_bar = ColorBar(color_mapper=color_mapper, major_label_text_font_size="12px",
                     ticker=tickers.AdaptiveTicker(desired_num_ticks=8),
                     formatter=PrintfTickFormatter(format="$ %d B"),
                     border_line_color=None,label_standoff=14,
                     location=(0, 0))
# load map for coloring
gdp_json = gdp_of_year(2019)
world_map = GeoJSONDataSource(geojson=gdp_json)
current_year = 2019
# initialize plotting figure
p=figure(title="{} World GDP Map".format(current_year), plot_height=640, plot_width=960)
p.grid.grid_line_color=None
p.axis.axis_line_color=None
p.axis.major_tick_line_color=None

g = p.patches("xs", "ys", source=world_map, line_color="black", line_width=0.3, 
          fill_color={"field": "gdp", "transform": color_mapper})
p.add_layout(color_bar, "right")
# unrecognized countries or regions
unid = Label(x=-190, y=-55, text="* white part: unidentified countries", text_font_size="12px")
p.add_layout(unid)

# add hover function to display country name and gdp when mouse is put on specific country
hover = HoverTool(renderers=[g], tooltips=[("Country/Region", "@country"), ("GDP", "@gdp{(0,0.00)}")])
p.add_tools(hover)

# add slider function to change the year to display
slider = Slider(start=min_year, end=max_year, value=current_year, step=1, title="YEAR")
def slider_callback(attr, old, new):
    # renew data used for each year
    new_gdp = gdp_of_year(new)
    world_map.geojson = new_gdp
    p.title.text = "{} World GDP Map".format(new)
slider.on_change('value', slider_callback)
layout = column(slider, p)

output_notebook()
show(layout)
# output_file("gdp.html")