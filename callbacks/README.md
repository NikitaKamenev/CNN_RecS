<h1>Классы обратного вызова для сбора метрик</h1>
<div class="mt-4">
<h2>Документация по использованию</h2> 
<h3>Шаг 1: Инициализация класса метрики с необходимыми параметрами</h3>
<h3>Шаг 2: Передача инициализированного класса в качестве параметра callback</h3>
<p>Может быть передан в методе fit, evaluate, predict</p>

<h3>Пример использования всех классов</h3>
<pre>
<code>token = 'YOUR-TOKEN'
base_url = 'http://localhost:5000/'
my_callbacks = [
    TimeCallback(),
    WeightsAverageChange(url=f"{base_url}/WeightsAverageChange", token=token),
    ActivationAverageChange(test_data=test_images, ratio=0.1, url=f"{base_url}/ActivationAverageChange", token=token),
    ActivationAverageValue(test_data=test_images, ratio=0.1,  url=f"{base_url}/ActivationAverageValue", token=token),
    GradientAverageValue(test_data=test_images, test_label=test_labels, ratio=0.1, url=f"{base_url}/GradientAverageValue", token=token)
]
model.fit(train_data, train_labels, epochs=10, callbacks=[my_callbacks])</code>
</pre>