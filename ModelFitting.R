library(fpp3)
library(readxl)
library(ggplot2)
library(patchwork) # for 'plot1/plot2' syntax
library(fabletools)
library(distributional) # For handling distributions
library(dplyr)
library(kableExtra) # convert table into latex format
library(clipr) # copy string from R quickly
library(fable)


# read the data into R environment
train_data_path <- "/Users/gufeng/2024_Fall/dasc6510/Final report for time series/data/train_data.csv"
train_data = read.csv(train_data_path)
train_data <- train_data |> 
  as_tibble() |>
  mutate(date = as.Date(date)) |>
  relocate(date, .before = meantemp) |>
  as_tsibble(index = date)

test_data_path <- "/Users/gufeng/2024_Fall/dasc6510/Final report for time series/data/test_data.csv"
test_data = read.csv(test_data_path)
test_data <- test_data |> 
  as_tibble() |>
  mutate(date = as.Date(date)) |>
  relocate(date, .before = meantemp) |>
  as_tsibble(index = date)

# doing time series decomposition on the mean temperature to check its trend and seasonality
# train_data |> 
decomposition_plot <- train_data |> 
  model(
  STL(meantemp ~ trend(60) + season(365),
      robust = TRUE)) |>
  components() |>
  autoplot()

# image storing path (one more '/' at the end)
image_path <-  "/Users/gufeng/2024_Fall/dasc6510/Final report for time series/documents/images/"

# store the image
ggsave(paste(image_path, "/decomposition_plot.png", sep = ''),
       plot = decomposition_plot,
       width = 8, 
       height = 5)

# define a function to quickly save plots
save_plot <- function(plot_, figure_name, width = 8, height = 5) {
  ggsave(
  paste(image_path, figure_name, sep = ''),
  plot = plot_,
  width = width, 
  height = height
  )
}

# build the standard linear model and four benchmarks
benchmarks <- train_data |> 
  model(
    linear_no_dummy = 
      TSLM(meantemp ~ humidity + wind_speed + meanpressure + time),
    linear_dummy = 
      TSLM(meantemp ~ humidity + wind_speed + meanpressure + time + 
             season_Autumn + season_Spring + season_Summer),
    naive = NAIVE(meantemp),
    snaive = SNAIVE(meantemp), # the seasonal period is not correctly spcified here
    drift = RW(meantemp ~ drift())
  )

# Define a function to quickly get the latex code for a table from R
to_latex_tabular <- function(tabular) {
  # Convert the table to LaTeX code
  latex_tabular <- tabular |> 
    mutate(across(where(is.numeric), ~ round(.x, 4)))|> 
    kbl(format = "latex", booktabs = TRUE, caption = "Model Summary Table") |>
    kable_styling(latex_options = c("hold_position"))
  # copy the LaTeX code to Clipboard
  write_clip(latex_tabular)
}
  
# Generate the glance result
glance_benchmarks <- glance(benchmarks) |> 
  mutate(across(where(is.numeric), ~ round(.x, 4)))

to_latex_tabular(glance_benchmarks)



# create an SARIMA-improved linear model
sarima_model <- train_data |> 
  model(
    # search a proper SARIMA model but confine 
    # the number of parameters of SARIMA part
    sarima = ARIMA(
      formula = meantemp ~ 
        humidity + wind_speed + meanpressure + time + 
        season_Autumn + season_Spring + season_Summer +
        pdq(
          p = 0:2, 
          d = 0:2,
          q = 0:2,
          p_init = 0, q_init = 0, fixed = list()
        ) + 
        PDQ(
          P = 0:2, 
          D = 0:2, 
          Q = 0:2,
          period = 365, # observed and inferred from the STL decomposition and empirial
          P_init = 0, Q_init = 0, fixed = list()
          ),
      stepwise = TRUE
    )
  )

# stack all these models together and compare them
all_models <- bind_cols(benchmarks, sarima_model)

fitted_values <- all_models |> 
  augment()

# draw the fitted values and store
fitted_value_plot <- fitted_values |> 
  autoplot(.fitted) + 
  geom_line(mapping = aes(x=date, y = meantemp, color = 'ture value', alpha = .1), 
            data = fitted_values) +
  facet_wrap(vars(.model), scales = "free_y", ncol = 2)  +
  scale_color_manual(values = c('meantemp' = 'black')) +  # Explicitly set color for True Value
  theme_minimal()+
  theme(legend.position = 'none')

# save the plot
save_plot(fitted_value_plot, "fitted_values.png")

# doing stationary tests on residuals of these models
resid_tests <- fitted_values |> 
  features(
    .resid,
    c(unitroot_kpss,
      unitroot_ndiffs,
      unitroot_nsdiffs,
      box_pierce,
      ljung_box
      ))
to_latex_tabular(resid_tests)

# 



