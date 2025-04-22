# check carefully if my codes have any bug or if I have done anything unexpected/unreasonable
# check my comments. there are some issues that you need to address

#######  read the IG score results  ############

ig_b=read.csv("bcell10.csv",stringsAsFactors = F)
ig_m=read.csv("macrophage7.csv",stringsAsFactors = F)
ig_e=read.csv("endothelial4.csv",stringsAsFactors = F)
ig_f=read.csv("fibroblast10.csv",stringsAsFactors = F)
ig_t=read.csv("tcell9.csv",stringsAsFactors = F)

ig_b=ig_b[order(ig_b$Gene),]
ig_m=ig_m[order(ig_m$Gene),]
ig_e=ig_e[order(ig_e$Gene),]
ig_f=ig_f[order(ig_f$Gene),]
ig_t=ig_t[order(ig_t$Gene),]

ig=data.frame(gene=ig_m$Gene,
              b=rank(-ig_b$IG.Score.After.Training),
              m=rank(-ig_m$IG.Score.After.Training),
              e=rank(-ig_e$IG.Score.After.Training),
              f=rank(-ig_f$IG.Score.After.Training),
              t=rank(-ig_t$IG.Score.After.Training)
)

ig=ig[abs(ig_m$IG.Score.After.Training)>0.000001,]
cor(ig[,-1])

tmp=ig
  
for (i in 2:dim(ig)[2])
{
  keep=ig[,i]<=apply(tmp[,-c(1,i)],1,median)
  ig[!keep,i]=Inf # this step of contrasting is optional, maybe better not use this filtering
  write.table(ig[order(ig[,i]),1],file=paste(colnames(ig)[i],".txt",sep=""),
              row.names = F,col.names = F,quote=F)
    # submit to gorilla GO analysis
}
  
ig=ig[ig$t!=Inf,] # this step of contrasting is optional, maybe better not use this filtering

#########  compare ig score with HLA binder counts  ##########

# hla binder count
# the data were curated from Jia's output:
# sh /project/DPDS/Wang_lab/shared/spatial_TCR/code/netmhcoutput/command.sh
netmhc=read.table("clean.txt",
             stringsAsFactors = F,sep="\t")
netmhc$V2=sapply(strsplit(netmhc$V2,split="_"),function(x) x[2]) # get protein name
netmhc=netmhc[netmhc$V3<=0.05,] 
netmhc=aggregate(netmhc$V1,by=list(netmhc$V2),length) # how many binders per protein/gene

# protein id <-> gene name mapping
id=read.csv("idmapping_stagei.csv",
            stringsAsFactors = F)
# a very small number of genes share the same protein name
# why? need to figure out
# but this is not a major issue
id=id[!duplicated(id$Protein),] 
rownames(id)=id$Protein
netmhc$gene_name=id[netmhc$Group.1,"Gene"]

# finding the total number of possible epitopes
# the data are in stagei_hla_binder, this is a temporary solution
# better to merge the total epitope count data with the idmapping_stagei.csv file to be cleaner
tmp=read.csv("stagei_hla_binder.csv",stringsAsFactors = F)

# add HLA binder count and total epitope count to the ig file
ig$count=0
ig$total=0
for (i in 1:dim(ig)[1]) 
{
  if (ig$gene[i] %in% netmhc$gene_name) 
  {
    ig$count[i]=netmhc$x[netmhc$gene_name==ig$gene[i]]
  }
  
  # note that a few genes in the stagei_hla_binder.csv file has empty information
  # on the total number of epitopes. Like the XIST gene
  # why? feels like a bug. need to correct
  # but this is not a major issue
  ig$total[i]=tmp$peptides[tmp$Gene==ig$gene[i]]
}

# plotting
ig=ig[order(ig$t),]
# top cutoff*100% of genes with the highest ig scores
cutoffs=c(0.002,0.003,0.005,0.01,0.02,0.05,
          0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
indices=1:dim(ig)[1]
statistics=log((ig$count+1)/ig$total)

plot(cutoffs,
  sapply(cutoffs,function(cutoff) median(statistics[indices<=cutoff*dim(ig)[1]],na.rm=T)),
  xlab="top % of genes with largest T ig scores",
  ylab="normalized binder counts")


library(ggplot2)
library(viridis)  # for perceptually uniform color scale

# Prepare the data
cutoffs = c(0.002, 0.003, 0.005, 0.01, 0.02, 0.05,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
ig = ig[order(ig$t), ]  # rank genes by T cell IG scores
statistics = log((ig$count + 1) / ig$total)  # log-normalized HLA binder score

# Calculate median binder score for each top cutoff
plot_df = data.frame(
  Cutoff = cutoffs,
  MedianBinderScore = sapply(cutoffs, function(cutoff) {
    top_n = floor(cutoff * nrow(ig))
    median(statistics[1:top_n], na.rm = TRUE)
  })
)

# Assign color to each cutoff point using the viridis color palette
plot_df$color = viridis::viridis(length(cutoffs))

# Generate the plot
ggplot(plot_df, aes(x = Cutoff, y = MedianBinderScore)) +
  # Draw a thin blue line connecting all points
  geom_line(color = "#1f78b4", size = 0.8) +
  
  # Add colorful points with gradient color based on cutoff percentiles
  geom_point(aes(color = Cutoff), size = 2) +
  
  # Use viridis color scale for better aesthetics and colorblind-friendliness
  scale_color_viridis_c(option = "D", direction = -1) +
  
  # Use a clean, minimal theme
  theme_minimal(base_size = 14) +
  
  # Set axis labels and plot title
  labs(
    x = "Top % of Genes (Ranked by T Cell IG Score)",
    y = "Log(Normalized HLA Binder Counts)",
    title = "NetMHCHLA Epitope Presentation",
    color = "Gene Rank Percentile"
  ) +
  
  # Customize text and layout for better readability
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(color = "black"),
    legend.position = "right"
  )

main_col   <- "#4A6FA5"                                 # deep gray‑blue for lines/points
accent_pal <- c("#E0E0E0", "#B0B8C4", "#708090") 

theme_journal <- theme_classic(base_size = 19) +        # slightly larger base font
  theme(
    plot.title      = element_text(face = "bold", hjust = 0.5, size = 22),
    axis.title      = element_text(face = "bold", size = 19),
    axis.text       = element_text(color = "black", size = 17),
    axis.line       = element_line(color = "grey50", linewidth = 0.6),
    axis.ticks      = element_line(color = "grey50"),
    legend.title    = element_text(face = "bold", size = 17),
    legend.text     = element_text(size = 15),
    panel.spacing   = unit(0.3, "lines"),               # minimal gap between panels
    plot.margin     = margin(3, 3, 3, 3),               # 3‑pt white space around the plot
    panel.grid.major = element_blank(),                 # remove major grid lines
    panel.grid.minor = element_blank()
  )

## ───────── 3. Panel A: Immune gradient vs HLA epitope ─────────
pA <- ggplot(plot_df, aes(Cutoff, MedianBinderScore)) +
  geom_line(color = main_col, linewidth = 1.4) +
  geom_point(aes(color = Cutoff), size = 3) +
  scale_color_gradientn(
    colors = accent_pal,
    name   = "Gene Rank\nPercentile"
  ) +
  labs(
    x = "Top % of Genes",
    #y = expression(log[10]~"(Normalized HLA Binder Counts)"),
    y = "Log(normalized HLA binder counts)",
  ) +
  theme_journal +
  theme(
    axis.text.x     = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

print(pA)
ggsave(
  "netmhc.pdf",
  pA,
  width  = 7, height = 6, units = "in",
  dpi    = 300,
  useDingbats = FALSE                                   # avoid font‑embedding issues
)
