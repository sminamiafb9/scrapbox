CREATE OR REPLACE FUNCTION sandbox.shotaro_minami8e40a5.greet(name STRING)
RETURNS STRING
LANGUAGE PYTHON
DETERMINISTIC
ENVIRONMENT (
  environment_version = '6'
)
AS $$
greeting_prefix = "Hello"
return f"{greeting_prefix}, {name}!"
$$;