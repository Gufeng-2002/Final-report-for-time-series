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
  as_tsibble(index = date) |> 
  mutate(season_Autumn = 0, season_Summer = 0) |> 
  filter(X != 0)

# doing time series decomposition on the mean temperature to check its trend and seasonality
# train_data |> 
decomposition_plot <- train_data |> 
  model(
  STL(meantemp ~ trend(60) + season(365),
      robust = TRUE)) |>
  components() |>
  autoplot() + 
  theme_minimal()

# image storing path (one more '/' at the end)
image_path <-  "/Users/gufeng/2024_Fall/dasc6510/Final report for time series/documents/images/"

# store the image
# ggsave(paste(image_path, "decomposition_plot.png", sep = ''),
#        plot = decomposition_plot,
#        width = 8, 
#        height = 5)

# define a function to quickly save plots
save_plot <- function(plot_, figure_name, 
                      store_path = image_path, width = 8, height = 5) {
  ggsave(
  paste(store_path, figure_name, sep = ''),
  plot = plot_,
  width = width, 
  height = height
  )
}

# save the docomposition plot
# save_plot(decomposition_plot, "decomposition_plot.png")

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
    drift = RW(meantemp ~ drift()),
    mean = MEAN(meantemp)
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

# create an SARIMA-improved linear model
sarima_model <- train_data |> 
  model(
    # search a proper SARIMA model but confine 
    # the number of parameters of SARIMA part
    sarima_dummy = ARIMA(
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
    ),
    sarima_no_dummy = ARIMA(
      formula = meantemp ~ 
        humidity + wind_speed + meanpressure + time + 
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
fitted_values |> 
  autoplot(.fitted) + 
  facet_wrap(vars(.model), scales = "free_y", ncol = 2)  +
  geom_line(mapping = aes(x=date, y = meantemp),
            color = 'black',
            alpha = .5,
            data = fitted_values) +
  theme_minimal()+
  theme(legend.position = 'none')

# save the plot
save_plot(fitted_value_plot, "fitted_values.png")

# doing stationary tests on residuals of these models
resid_tests <- fitted_values |> 
  features(
    .resid,
    c(unitroot_kpss,
      box_pierce,
      ljung_box
      ))
to_latex_tabular(resid_tests)

# have a look at the glance function
all_models_glance <- glance(all_models) |>
  select(.model, adj_r_squared, sigma2, log_lik, AICc, BIC, df.residual)
to_latex_tabular(all_models_glance)

# have a look at the coefficients and generate the latex table
coeffs_all_models <- all_models |> coef()
to_latex_tabular(coeffs_all_models)

# doing forecasts with different models and comparing them
forecasts <- all_models |> forecast(test_data)

# plot the forecasts and save
forecasts_plot <- forecasts |> 
  autoplot(level = 90) + 
  facet_wrap(vars(.model), ncol = 2, scale = "free_y") +
  geom_line(
    data = test_data, 
    mapping = aes(x = date, y = meantemp))  + 
  theme_minimal()+ 
  theme(legend.position = "none")

# save the plot
# save_plot(forecasts_plot) # can not save the mininal theme, i do not why

# check the accuracy by checking some criteria
accuracies <- accuracy(forecasts, test_data) |> 
  select(-.type, -ME, -MASE, -RMSSE) |> 
  arrange(RMSE)

# generate latex table
to_latex_tabular(accuracies)

# summarize the information for sarima_dummy model(the best one)
to_latex_tabular(best_model |> coef())
best_forecast <- best_model |> forecast(test_data)
to_latex_tabular(accuracy(best_forecast, test_data))
to_latex_tabular(glance(best_model) |> 
  select(-ar_roots, -ma_roots)
  )
to_latex_tabular(best_model |> 
     augment() |> 
     features(.resid, c(ljung_box, box_pierce)))

# recheck the mean temperature for different seasons
temp_season_group <- train_data |> 
  as_tibble() |> 
  group_by(season) |> 
  summarise(mean_temp = mean(meantemp),
            mean_pressure = mean(meanpressure),
            mean_w_speed = mean(wind_speed),
            mean_humidity = mean(humidity))
to_latex_tabular(temp_season_group)

# add the residual diagnonistics plot





