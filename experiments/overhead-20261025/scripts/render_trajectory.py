"""Standalone trajectory figure and explicit missing-position CSV."""
import csv
import numpy as np

def render(replay,report,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyBboxPatch
    pid=replay['default_player'];fps=replay['fps'];observations=[];rows=[]
    for row in replay['frames']:
        o=next((o for o in row['objects']if o['player_id']==pid),None)
        xy=o['xy']if o else None
        if xy is not None:observations.append({'frame':row['frame'],**o})
        rows.append([row['frame']/fps,replay['start_s']+row['frame']/fps,pid,o['track_id']if o else'',*(xy if xy else['','']),'mapped'if xy else'visible_position_unavailable'if o else'not_tracked'])
    with (folder/'player-trajectory.csv').open('w',newline='')as stream:
        writer=csv.writer(stream);writer.writerow(['clip_seconds','recording_seconds','player_id','track_id','x_relative','y_relative','status']);writer.writerows(rows)
    fig,(ax,timeax)=plt.subplots(2,1,figsize=(13,8),gridspec_kw={'height_ratios':[8,1]},layout='constrained')
    fig.suptitle(report['selected_player']+' · five-minute player trajectory',fontsize=19,x=.04,ha='left')
    ax.set_title('NBHL · recording 15:00–20:00 · observed positions only',loc='left',fontsize=12,pad=15)
    ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle='round,pad=0,rounding_size=.06',edgecolor='#555',facecolor='#fafafa',linewidth=1.5))
    for key,color in [('center','#b65a5a'),('left_blue','#6f8dad'),('right_blue','#6f8dad')]:ax.axvline(replay['rink'][key],color=color,linewidth=1,alpha=.7)
    segments=[];times=[]
    for a,b in zip(observations,observations[1:]):
        if b['frame']==a['frame']+1 and a['segment']==b['segment']:
            segments.append([a['xy'],b['xy']]);times.append(b['frame']/fps)
    lc=LineCollection(segments,array=np.array(times),cmap='plasma',norm=plt.Normalize(0,300),linewidths=1.7,alpha=.85);ax.add_collection(lc)
    for o,label in [(observations[0],'First observed'),(observations[-1],'Last observed')]:
        ax.scatter(*o['xy'],c=[o['frame']/fps],cmap='plasma',vmin=0,vmax=300,s=50,edgecolor='#111',zorder=5)
        ax.annotate(label,xy=o['xy'],xytext=(7,10),textcoords='offset points',fontsize=10)
    ax.set(xlim=(-.03,1.03),ylim=(1.04,-.04));ax.set_aspect(.5);ax.axis('off')
    colorbar=fig.colorbar(lc,ax=ax,orientation='horizontal',fraction=.045,pad=.05,ticks=[0,60,120,180,240,300]);colorbar.ax.set_xticklabels(['0:00','1:00','2:00','3:00','4:00','5:00']);colorbar.set_label('Elapsed recording time')
    coverage=np.zeros(4500)
    for o in observations:coverage[o['frame']]=1
    timeax.imshow(coverage[None,:],extent=[0,300,0,1],aspect='auto',cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
    timeax.set(yticks=[],xticks=[0,60,120,180,240,300]);timeax.set_xticklabels(['0:00','1:00','2:00','3:00','4:00','5:00']);timeax.set_title(f'{len(observations)/fps:.1f}s of accepted floor positions / 300s recording · gaps remain empty',loc='left',fontsize=11)
    fig.text(.04,-.015,'Assistant-reviewed identity fragments. Approximate rink geometry; no measured distance, speed or total time on floor.',fontsize=10,color='#555')
    fig.savefig(folder/'player-trajectory.png',dpi=150,bbox_inches='tight');plt.close(fig)
