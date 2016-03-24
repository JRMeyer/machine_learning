import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        # return '{p:.2f}%  ({v:d} languages)'.format(p=pct,v=val)
        return '                         {v:d} languages'.format(v=val)
    return my_autopct

# The slices will be ordered and plotted counter-clockwise.
labels = '', ''
values = [21,3150]
colors = ['yellowgreen', 'lightcoral']
explode = (0.1,0)  # only "explode" the 2nd slice (i.e. 'Hogs')


fig = plt.figure()
fig.set_size_inches(8, 8)

ax = fig.gca()

patches, texts, autotexts = plt.pie(values, 
                                    explode=explode, 
                                    labels=labels, 
                                    colors=colors,
                                    autopct=make_autopct(values), 
                                    shadow=True, 
                                    startangle=115)

proptease = fm.FontProperties()
proptease.set_size('xx-large')
plt.setp(autotexts, fontproperties=proptease)
plt.setp(texts, fontproperties=proptease)

plt.title('Languages with > 10,000 Speakers',
          bbox={'facecolor':'0.8', 'pad':10}, 
          fontsize=24)


# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')

plt.savefig('out.png', transparent=True)

plt.show()

