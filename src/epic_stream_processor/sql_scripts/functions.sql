
create or replace function array_add(p_one double precision[], p_two double precision[])
  returns double precision[]
as
$$
declare
  l_idx int;
  l_result double precision[];
begin
  if p_one is null or p_two is null then
    return coalesce(p_one, p_two);
  end if;
  
  for l_idx in 1..greatest(cardinality(p_one), cardinality(p_two)) loop
    l_result[l_idx] := coalesce(p_one[l_idx],0) + coalesce(p_two[l_idx], 0);
  end loop;
  
  return l_result;  
end;  
$$
language plpgsql
immutable;

create aggregate array_element_sum(double precision[]) (
  sfunc = array_add,
  stype = double precision[],
  initcond = '{}'
);
