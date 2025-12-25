# Запуск и тестирование

## Сборка и запуск контейнеров

Для удобства был создан `Makefile`, в котором есть 3 таргета:
* build
* run
* stop

Для сборки и запуска нужно использовать:
```
make build && make run
```

Для остановки контейнеров:
```
make stop
```

## Тестирование системы

Первым делом необходимо зарегестрироваться, залогиниться и получить `Bearer` токен. Он необходим для выполнения *POST* запроса `/search/forward` и *DELETE* запроса `/search/history`.

Для регистрации необходимо сходить по `auth/register` и в теле запроса передать `JSON`:
```json
{
  "username": "username",
  "password": "password"

}
```

Далее логин по `/auth/login`. В теле точно такой же `JSON`:
```json
{
  "username": "username",
  "password": "password"

}
```

В ответ придет `Token`, который в дальнейшем должен использоваться как `Bearer Token` для доступа к ручкам `/search/forward` и `/search/history`.

```json
{
    "access_token": "token",
    "token_type": "bearer"
}
```


Теперь можно попробовать сходить к `/search/forward`. В теле запроса укажите следующий `JSON`:
```json
{
    "instruments": ["synthesizer", "drums"],
    "genres": ["pop", "dance", "synthpop"],
    "tags": ["energetic", "acoustic", "vocal", "voice"]
}
```

Можно посмотреть историю и статистику запросов при помощи `GET` запросов `/search/history` и `/search/stats` - для этих эндпоинтов `Bearer` авторизация *не обязательна*.

Для того, чтобы отправить `DELETE` запрос на эндпоинт `/search/history`, необходимо вручную залезть в базу и Вашему созданному пользователю в поле `is_admin` выставить `True`. Тогда у вас появятся права на удаление истории.
