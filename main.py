import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mlxtend.frequent_patterns import association_rules, apriori

data=pd.read_csv("Bakery.csv")
print('\nOut 1:\n',data)
# Checking the information and missing value of the data
print('\nOut 2:\n')
data.info()
print('\nOut 3:\n',len(data))
# The lenth of data is equal to number of non-null values, double check if there is null value
print('\nOut 4:\n',data.isnull().sum())
# We have one variable as integer which is transaction id. Other ones are strings. This mean that there is no null value in the dataset.
sns.heatmap(data.isnull())
plt.show()
# And for a better showcase, we correct the title fitting the spelling rule.
data.rename(columns={"DataTime":"Date Time","Daypart":"Day Part","DayType":"Day Type"},inplace=True)
print('\nOut 5:\n',data)
# Splitting the Date Time to hour,day,month and year
data["Year"]=pd.to_datetime(data["DateTime"]).dt.year
data["Month"]=pd.to_datetime(data["DateTime"]).dt.month
data["Week Day"]=pd.to_datetime(data["DateTime"]).dt.weekday
data["Hour"]=pd.to_datetime(data["DateTime"]).dt.hour
# In order to visualize better, we split the date object into the 4 parts. We will examine hour,day, month and year features seperately. Time division is adjusted to be show clearly what sepecific time period is.
data["Month"]=data["Month"].replace((1,2,3,4,5,6,7,8,9,10,11,12),('January','February' ,'March' ,'April' ,'May' ,'June' ,'July' ,'August' ,'September','October' ,'November' ,'December' ))
data["Week Day"]=data["Week Day"].replace((0,1,2,3,4,5,6),('Monday','Tuesday' ,'Wednesday' ,'Thursday','Friday' ,'Saturday' ,'Sunday'))
data["Hour"]=data["Hour"].replace((1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),('1-2','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18','18-19','19-20','20-21','21-22','22-23','23-24'))
print('\nOut 6:\n',data)
# Dropping the unnecessary feature and look to the new data
data = data.drop("DateTime", axis=1)
print('\nOut 7:\n',data.head())
# To begin with, we shows the basic data info first. The total number of transactions is:
print('\nOut 8:\n',len(data["TransactionNo"].value_counts()))
# start with visualizing the number of items. We will plot a bar plot to visualize the most 15 popular items
products=data["Items"].value_counts().head(15).reset_index(name="Count")
products=products.rename(columns={"index":"Items"})
plt.figure(figsize=(20,12))
ax=sns.barplot(x="Items",y="Count",data=products,palette='Greens_r')
for i in ax.containers:
    ax.bar_label(i)
    plt.title("15 Best-Selling Products",size=18, fontweight='bold')
    plt.xlabel('Items', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
# Sales by years
datayears=data.groupby("Year")["TransactionNo"].count().reset_index()
print('\nOut 9:\n',datayears)
plt.figure(figsize=(6,5))
ax=sns.barplot(x="Year",y="TransactionNo",data=datayears,palette='Paired')
for i in ax.containers:
    ax.bar_label(i)
    plt.title("Sales by years",size=18)
year = dict(data.groupby("Year")["Items"].count().sort_values(ascending=False))
plt.figure(figsize=(6,6))
plt.pie(year.values(), labels=year.keys(), explode = [0, 0.01], colors = sns.color_palette("Set3",8)[5:3:-1], autopct='%.2f%%')
plt.title("Items by Day Type", fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
# Sales by Month
datamonth=data.groupby("Month")["TransactionNo"].count().reset_index()
print('\nOut 10:\n',datamonth)
plt.figure(figsize=(15,5))
datamonth_sorted = datamonth.sort_values(by='TransactionNo', ascending=False)
ax=sns.barplot(x="Month",y="TransactionNo",data=datamonth_sorted,palette="Reds_r")
for i in ax.containers:
    ax.bar_label(i)
    plt.title("Sales by Month",size=18, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Trabsaction No.', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
# Sales by Week Day
dataweek=data.groupby("Week Day")["TransactionNo"].count().reset_index()
print('\nOut 11:\n',dataweek)
daytype = dict(data.groupby("Day Type")["Items"].count().sort_values(ascending=False))
# For a more detailed sales graph:
plt.figure(figsize=(6,6))
plt.pie(daytype.values(), labels=daytype.keys(), explode = [0, 0.01], colors = sns.color_palette("Set2")[3:5], autopct='%.2f%%')
plt.title("Items by Day Type", fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
plt.figure(figsize=(12,6))
dataweek_sorted = dataweek.sort_values(by='TransactionNo', ascending=False)
ax=sns.barplot(x="Week Day",y="TransactionNo",data=dataweek_sorted,palette='hot')
for i in ax.containers:
    plt.title("Sales by Week Day",size=18, fontweight='bold')
    plt.xlabel('Week Day', fontsize=14)
    plt.ylabel('Trabsaction No.', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
# Sales by Hour
datahour=data.groupby("Hour")["TransactionNo"].count().reset_index()
print('\nOut 12:\n',datahour)
plt.figure(figsize=(16,10))
datahour_sorted = datahour.sort_values(by='TransactionNo', ascending=False)
ax=sns.barplot(x="TransactionNo",y="Hour",data=datahour_sorted,palette='icefire')
for i in ax.containers:
    plt.title("Sales by Hour",size=18, fontweight='bold')
    plt.xlabel('Trabsaction No.', fontsize=14)
    plt.ylabel('Hour', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
plt.show()

# Sales by Day Part
datapart=data.groupby("Day Part")["TransactionNo"].count().reset_index()
print('\nOut 13:\n',datapart)
plt.figure(figsize=(12, 6))
sns.set_style("ticks")
colors = sns.color_palette("Set2")
datapart_sorted = datapart.sort_values(by='TransactionNo', ascending=False)
ax = sns.barplot(x="Day Part", y="TransactionNo", data=datapart_sorted, palette=colors, alpha=0.8) # 绘制柱状图
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=12, padding=4, labels=[f'{int(v.get_height()):,}' for v in i])
plt.title("Sales by Day Part", size=18, fontweight='bold')
plt.xlabel('Day Part', fontsize=14)
plt.ylabel('Transaction Number', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.despine()
plt.show()
# Product sales by parts of the day
dataparts = data.groupby(["Day Part", "Items"])["TransactionNo"].count().reset_index().sort_values(["Day Part", "TransactionNo"], ascending=False)
dayss = ['Morning', 'Afternoon', 'Evening', 'Night']
plt.figure(figsize=(18, 8))
colors = sns.color_palette('Set2')
for i, j in enumerate(dayss):
    plt.subplot(2, 2, i+1)
    partsdata = dataparts[dataparts["Day Part"] == j].head(10)
    ax = sns.barplot(data=partsdata, x="TransactionNo", y="Items", palette=colors, alpha=0.8)
    for i in ax.containers:
        ax.bar_label(i, label_type='edge', fontsize=12, padding=4, labels=[f'{int(v.get_width()):,}' for v in i])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Customers care to buy these products at ' + "{}".format(j), size=18, fontweight='bold')  # 添加子图标题
plt.tight_layout()
sns.despine()
plt.show()

# Sales divide by the months
plotSize=(25,20)
showNum=15
labelSize=12
titleSize=18
tickSize=12
MonthMsg=['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']
tempData=data.groupby(["Month","Items"])
tempDataMsg=tempData["Items"].count()
ItemMsgInitial=tempDataMsg.reset_index(name="Quantity")
ItemMsg=ItemMsgInitial.sort_values(["Month","Quantity"],ascending=False)
plt.figure(figsize=plotSize)
plt.subplots_adjust(wspace =1, hspace =0.5)
plotColor=sns.color_palette("hls", showNum)
for num,month_data in enumerate(MonthMsg):
    plt.subplot(3,4,num+1)
    plt.xticks()
    plt.yticks()
    plt.title('Sales in {}'.format(month_data), size=titleSize, fontweight='bold')
    ItemMsgSelected=ItemMsg[ItemMsg["Month"]==month_data].head(showNum)
    BarPlot=sns.barplot(data=ItemMsgSelected,x="Quantity",y="Items",palette=plotColor)
    plt.xlabel('Quantity', size=labelSize, fontweight='bold')
    plt.ylabel('Items', size=labelSize, fontweight='bold')
    for num in BarPlot.containers:
        BarPlot.bar_label(num)
plt.show()
# Sales(Sum) divide by the months
tempData=data.groupby(["Month"])
tempDataMsg=tempData["Items"].count()
ItemMsgInitial=tempDataMsg.reset_index(name="Quantity")
ItemMsg=ItemMsgInitial.sort_values(["Quantity"],ascending=False)
plt.figure(figsize=plotSize)
plt.xlabel('Month', size=labelSize, fontweight='bold')
plt.ylabel('Quantity of Sale', size=labelSize, fontweight='bold')
plt.title('Sum Quantity in Different Months', size=titleSize, fontweight='bold')
BarPlot=sns.barplot(data=ItemMsg,x="Month",y="Quantity",palette=plotColor)
for num in BarPlot.containers:
    BarPlot.bar_label(num)
plt.show()
# Shown by the fan diagram
plt.pie(ItemMsg["Quantity"],labels=ItemMsg["Month"],autopct='%.2f%%',colors=plotColor,explode=[0.2,0,0,0,0,0,0,0,0,0,0,0])
plt.title("Ratio of Monthly sales", size=titleSize, fontweight='bold')
plt.show()
# Shown by the line chart
monthSequentialData=[1,2,3,4,5,6,7,8,9,10,11,12]
monthSequentialItemData=[0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(ItemMsg["Month"])):
    if ItemMsg["Month"][i] == 'January':
        monthSequentialItemData[0]=ItemMsg["Quantity"][i]
    elif ItemMsg["Month"][i] == 'February':
        monthSequentialItemData[1]=ItemMsg["Quantity"][i]
    elif ItemMsg["Month"][i] == 'March':
        monthSequentialItemData[2]=ItemMsg["Quantity"][i]
    elif ItemMsg["Month"][i] == 'April':
        monthSequentialItemData[3]=ItemMsg["Quantity"][i]
    elif ItemMsg["Month"][i] == 'May':
        monthSequentialItemData[4]=ItemMsg["Quantity"][i]
    elif ItemMsg["Month"][i] == 'June':
        monthSequentialItemData[5]=ItemMsg["Quantity"][i]
    elif ItemMsg["Month"][i] == 'July':
        monthSequentialItemData[6]=ItemMsg["Quantity"][i]
    elif ItemMsg["Month"][i] == 'August':
        monthSequentialItemData[7]=ItemMsg["Quantity"][i]
    elif ItemMsg["Month"][i] == 'September':
        monthSequentialItemData[8]=ItemMsg["Quantity"][i]
    elif ItemMsg["Month"][i] == 'October':
        monthSequentialItemData[9]=ItemMsg["Quantity"][i]
    elif ItemMsg["Month"][i] == 'November':
        monthSequentialItemData[10]=ItemMsg["Quantity"][i]
    elif ItemMsg["Month"][i] == 'December':
        monthSequentialItemData[11]=ItemMsg["Quantity"][i]
plt.xlabel('Month', size=labelSize, fontweight='bold')
plt.ylabel('Quantity of Sale', size=labelSize, fontweight='bold')
plt.title("Change of Monthly Sales", size=titleSize, fontweight='bold')
axis=plt.gca()
axis.xaxis.set_major_locator(MultipleLocator(1))
plt.plot(monthSequentialData,monthSequentialItemData,color='r')
plt.show()
# Sales divide by the quarters
data["Quarter"]=0
for i in range(len(data["Month"])):
    if data["Month"][i]=='January' or data["Month"][i]=='February' or data["Month"][i]=='March':
        data["Quarter"][i]='First Quarter'
    elif data["Month"][i]=='April' or data["Month"][i]=='May' or data["Month"][i]=='June':
        data["Quarter"][i]='Second Quarter'
    elif data["Month"][i]=='July' or data["Month"][i]=='August' or data["Month"][i]=='September':
        data["Quarter"][i]='Third Quarter'
    elif data["Month"][i]=='October' or data["Month"][i]=='November' or data["Month"][i]=='December':
        data["Quarter"][i]='Fourth Quarter'
QuarterMsg=['First Quarter','Second Quarter','Third Quarter','Fourth Quarter']
tempData=data.groupby(["Quarter","Items"])
tempDataMsg=tempData["Items"].count()
ItemMsgInitial=tempDataMsg.reset_index(name="Quantity")
ItemMsg=ItemMsgInitial.sort_values(["Quarter","Quantity"],ascending=False)
plt.figure(figsize=plotSize)
plt.subplots_adjust(wspace =0.5, hspace =0.5)
plotColor=sns.color_palette("hls", showNum)
for num,quarter_data in enumerate(QuarterMsg):
    plt.subplot(1,4,num+1)
    plt.xticks()
    plt.yticks()
    plt.title('{}'.format(quarter_data), size=titleSize, fontweight='bold')
    ItemMsgSelected=ItemMsg[ItemMsg["Quarter"]==quarter_data].head(showNum)
    BarPlot=sns.barplot(data=ItemMsgSelected,x="Quantity",y="Items",palette=plotColor)
    plt.xlabel('Quantity', size=labelSize, fontweight='bold')
    plt.ylabel('Items', size=labelSize, fontweight='bold')
    for num in BarPlot.containers:
        BarPlot.bar_label(num)
plt.show()
# Sales(Sum) divide by the quarters
tempData=data.groupby(["Quarter"])
tempDataMsg=tempData["Items"].count()
ItemMsgInitial=tempDataMsg.reset_index(name="Quantity")
ItemMsg=ItemMsgInitial.sort_values(["Quantity"],ascending=False)
plt.figure(figsize=plotSize)
plt.xlabel('Quarter', size=labelSize, fontweight='bold')
plt.ylabel('Quantity of Sale', size=labelSize, fontweight='bold')
plt.xticks(size=tickSize)
plt.yticks(size=tickSize)
plt.title('Sum Quantity in Different Quarters', size=titleSize, fontweight='bold')
BarPlot=sns.barplot(data=ItemMsg,x="Quarter",y="Quantity",palette=plotColor)
for num in BarPlot.containers:
    BarPlot.bar_label(num)
plt.show()
# Shown by the fan diagram
plt.pie(ItemMsg["Quantity"],labels=ItemMsg["Quarter"],autopct='%.2f%%',colors=plotColor)
plt.title("Ratio of Sales Per Quarter", size=titleSize, fontweight='bold')
plt.show()

dataapriori=data.groupby(["TransactionNo","Items"])["Items"].count().reset_index(name="Quantity")
datapivot=dataapriori.pivot_table(index="TransactionNo",columns="Items",values="Quantity",aggfunc="sum").fillna(0)
def table(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
datapivottable=datapivot.applymap(table)
print('\nOut 14:\n',datapivottable)
aprioridata=apriori(datapivottable,min_support=0.01,use_colnames=True)
print('\nOut 15:\n',aprioridata)
rules=association_rules(aprioridata, metric = "lift", min_threshold = 1)
print('\nOut 16:\n',rules.sort_values("confidence",ascending=False).head(10))
print('\nOut 17:\n',rules.sort_values("lift",ascending=False).head(10))