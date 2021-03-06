% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/autoDataPrep.R, R/printAutoDataPrep.R
\name{autoDataprep}
\alias{autoDataprep}
\alias{print.autoDataprep}
\title{Automatic data preparation for ML algorithms}
\usage{
autoDataprep(
  data,
  target = NULL,
  missimpute = "default",
  auto_mar = FALSE,
  mar_object = NULL,
  dummyvar = TRUE,
  char_var_limit = 12,
  aucv = 0.02,
  corr = 0.99,
  outlier_flag = FALSE,
  interaction_var = FALSE,
  frequent_var = FALSE,
  uid = NULL,
  onlykeep = NULL,
  drop = NULL,
  verbose = FALSE
)

\method{print}{autoDataprep}(x)
}
\arguments{
\item{data}{[data.frame | Required] dataframe or data.table}

\item{target}{[integer | Required] dependent variable (binary or multiclass)}

\item{missimpute}{[text | Optional] missing value impuation using mlr misimpute function. Please refer to the "details" section to know more}

\item{auto_mar}{[character | Optional] identify any missing variable which are completely missing at random or not (default FALSE). If TRUE this will call autoMAR()}

\item{mar_object}{[character | Optional] object created from autoMAR function}

\item{dummyvar}{[logical | Optional] categorical feature engineering i.e. one hot encoding (default is TRUE)}

\item{char_var_limit}{[integer | Optional] default limit is 12 for a dummy variable preparation. e.g. if gender variable has two different value "M" and "F", then gender has 2 levels}

\item{aucv}{[integer | Optional] cut off value for AUC based variable selection}

\item{corr}{[integer | Optional] cut off value for correlation based variable selection}

\item{outlier_flag}{[logical | Optional] to add outlier features (default is FALSE)}

\item{interaction_var}{[logical | Optional] bulk interactions transformer for numerical features}

\item{frequent_var}{[logical | Optional] frequent transformer for categorical features}

\item{uid}{[character | Optional] unique identifier column if any to keep in the final data set}

\item{onlykeep}{[character | Optional] only consider selected variables for data preparation}

\item{drop}{[character | Optional] exclude variables from the dataset}

\item{verbose}{[logical | Optional] display executions steps on console(default is FALSE)}

\item{x}{an object of class \code{\link{autoDataprep}}}
}
\value{
list output contains below objects

\describe{
  \item{\code{complete_data}}{complete dataset including new derived features based on the functional understanding of the dataset}
  \item{\code{master_data}}{filtered dataset based on the input parameters}
  \item{\code{final_var_list}}{list of master variables}
  \item{\code{auc_var}}{list of auc variables}
  \item{\code{cor_var}}{list of correlation variables}
  \item{\code{overall_var}}{all variables in the dataset}
  \item{\code{zerovariance}}{variables with zero variance in the dataset}
}
}
\description{
Final data preparation before ML algorithms. Function provides final data set and highlights of the data preparation
}
\details{
Missing imputation using impute function from MLR

MLR package have a appropriate way to impute missing value using multiple methods.
#' \itemize{
  \item mean value for integer variable
  \item median value for numeric variable
  \item mode value for character or factor variable
}
optional: You might be interested to impute missing variable using ML method. List of algorithms will be handle missing variables in MLR package
listLearners("classif", check.packages = TRUE, properties = "missings")[c("class", "package")]

Feature engineering
\itemize{
  \item missing not completely at random variable using autoMAR function
  \item date transfomer like year, month, quarter, week
  \item frequent transformer counts each categorical value in the dataset
  \item interaction transformer using multiplication
  \item one hot dummy coding for categorical value
  \item outlier flag and capping variable for numerical value
}

Feature reduction
\itemize{
  \item zero variance using nearZeroVar caret function
  \item pearson's correlation value
  \item auc with target variable
}
}
\section{Methods (by generic)}{
\itemize{
\item \code{print}: Print Method for the autoDataprep

Print the result of autoDataprep object
}}

\examples{
#Auto data prep
traindata <- autoDataprep(heart, target = "target_var", missimpute = "default",
dummyvar = TRUE, aucv = 0.02, corr = 0.98, outlier_flag = TRUE,
interaction_var = TRUE, frequent_var = TRUE)
train <- traindata$master
}
\seealso{
\code{\link[mlr:impute]{impute}}
}
\keyword{print}
