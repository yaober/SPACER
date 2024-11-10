# Jia, please try to understand my codes (why the data were processed in this way)
# please also help me check to make sure there are no bugs (make sure the processing makes sense to you)

#########  identify genes that have higher expression in tumor vs normal tissues in TCGA  ####
# so here, these analyses were performed at the bulk tissue level

setwd("/project/DPDS/Wang_lab/shared/BCR_antigen/data/TCGA")

cancers=c("UVM","UCS","UCEC","THYM","THCA","TGCT","STES","STAD",
  "SKCM","SARC","READ","PRAD","PCPG","PAAD","OV","MESO",
  "LUSC","LUAD","LIHC","LGG","LAML","KIRP","KIRC","KICH","HNSC",
  "GBM","ESCA","DLBC","COAD","CHOL","CESC","BRCA","BLCA","ACC")

all_exp_mat=NULL

for (cancer in cancers)
{
  print(cancer)
  file=paste(cancer,"/",cancer,".txt",sep="")
  
  # read in exp data for each cancer
  exp_mat=read.table(file,stringsAsFactors = F,header=T,sep="\t",row.names = 1)
  exp_mat=as.matrix(exp_mat)
  exp_mat=exp_mat[-1,]
  exp_mat_tmp=matrix(as.numeric(exp_mat),ncol=dim(exp_mat)[2])
  rownames(exp_mat_tmp)=rownames(exp_mat)
  colnames(exp_mat_tmp)=colnames(exp_mat)
  exp_mat=exp_mat_tmp
  
  # simple log+library size normalization
  exp_mat=log(exp_mat+1)
  exp_mat=t(t(exp_mat)-colMeans(exp_mat))
  
  # segregate into tumor vs normal samples
  tumor_label=as.numeric(substr(colnames(exp_mat),14,15))<=9
  exp_mat=aggregate(t(exp_mat),by=list(tumor_label),mean)
  rownames(exp_mat)=paste(cancer,exp_mat$Group.1)
  exp_mat=exp_mat[,-1]
  
  # aggregate isoform level exp data to gene level
  gene_aggregate=sapply(strsplit(colnames(exp_mat),"\\|"),function(x) x[1])
  exp_mat=aggregate(t(as.matrix(exp_mat,nrow=dim(exp_mat)[1])),by=list(gene_aggregate),mean)
  exp_mat=exp_mat[order(exp_mat$Group.1),]
  rownames(exp_mat)=exp_mat[,1]
  exp_mat=exp_mat[,-1,drop=F]
  
  # combine data for each cancer type
  if (is.null(all_exp_mat))
  {
    all_exp_mat=exp_mat    
  }else
  {
    exp_mat=exp_mat[rownames(exp_mat) %in% rownames(all_exp_mat),,drop=F]
    if (any(rownames(all_exp_mat)!=rownames(exp_mat))) {stop("rownames mistmatches")}
    all_exp_mat=cbind(all_exp_mat,exp_mat)  
  }
}

save(all_exp_mat,file="/project/DPDS/Wang_lab/shared/spatial_TCR/data/tumor_specific_genes/temp.RData")
all_exp_mat_tumor=rowMeans(all_exp_mat[,grepl("TRUE",colnames(all_exp_mat))])
all_exp_mat_normal=rowMeans(all_exp_mat[,!grepl("TRUE",colnames(all_exp_mat))])

tumor_antigens=rownames(all_exp_mat)[(all_exp_mat_tumor>all_exp_mat_normal+0.0001) & 
    all_exp_mat_normal<2]
length(tumor_antigens)

#########  identify genes that have minimum expression in each normal cell type  ####
# Human Protein Atlas data
# so here, these analyses were performed at the single cell type level
# genes that are expressed highly in any of the normal cell type won't become an antigen
# even genes that are expressed in normal epithelial cells (most tumors are from malignant epithelial cells)
# (exclude genes that happen early on in embryonic development)

hpa=read.table("/project/DPDS/Wang_lab/shared/spatial_TCR/data/tumor_specific_genes/rna_single_cell_type.tsv",
               stringsAsFactors = F,header=T,sep="\t")
hpa=hpa[!hpa$Cell.type %in% c("Spermatocytes","Early spermatids","Spermatogonia",
  "Late spermatids","Oocytes","Undifferentiated cells","Undifferentiated cells",
  "Extravillous trophoblasts","Cytotrophoblasts","Hofbauer cells","Erythroid cells",
  "Syncytiotrophoblasts"),] # genes that are highly expressed in early stage of human life
hpa$nTPM=log(hpa$nTPM+1)
hpa=aggregate(hpa$nTPM,by=list(hpa$Gene.name),function(x) quantile(x,0.95))
normal_expressed_genes=hpa$Group.1[hpa$x>5]

tumor_antigens=tumor_antigens[!tumor_antigens %in% normal_expressed_genes]
tumor_antigens=tumor_antigens[tumor_antigens!="?"]
length(tumor_antigens)

###########  intersect with dataset-specific tumor top genes in the SRT data  ######
# so in each SRT dataset, look at the top 1000 (?) genes that are highly expressed in the
# tumor cells of that SRT dataset
# then only look at genes that exist within "tumor_antigens"
# let's see how many genes are left in each SRT dataset after this filtering

write.table(tumor_antigens,file="/project/DPDS/Wang_lab/shared/spatial_TCR/data/tumor_specific_genes/tumor_antigens.txt",
            row.names = F,col.names = F,quote=F)
write.table(normal_expressed_genes, file="/project/DPDS/Wang_lab/shared/spatial_TCR/data/tumor_specific_genes/normal_expressed_genes.txt",
            row.names = F, col.names = F, quote = F)