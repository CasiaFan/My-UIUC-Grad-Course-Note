load expression_table.mat
gene_variation=std(exp_t')'; 
[a,b]=sort(gene_variation,'descend');
ngenes=100;
exp_t1=exp_t(b(1:ngenes),:);
gene_names1=gene_names(b(1:ngenes));
%%% for group 1
CGobj1 = clustergram(exp_t1, 'Standardize','Row',...
    'RowLabels', gene_names1,'ColumnLabels',array_names)
set(CGobj1,'RowLabels',gene_names1,'ColumnLabels'...
    ,array_names,'linkage', 'single','RowPDist','cityblock', 'Standardize', 'Row');
%%%
%Which biological functions are overrepresented in different clusters?
%1) Pick a cluster:
%2) Select a node on the tree of rows, 
%3) Right click
%4) Choose export group info into the workspace
%5) Name it gene_list
%Run the following two Matlab commands to display genes
g1=group3.RowLabels;
for m=1:length(g1); 
    disp(g1{m}); 
end;
%%
% copy the list of displayed genes 
% go to "Start Analysis" on https://david.ncifcrf.gov/tools.jsp
% paste gene list into the box in the left panel
% select ENSEMBL_GENE_ID
% select gene list radio button
% pick "Functional Annotation Clustering" for gene list broken into multiple groups (clusters) each with related biological functions
% Select "Functional Annotation Clustering" rectangular button below to display annotation results
% Write down the # of genes in the cluster and the top functions in two most interesting clusters 
