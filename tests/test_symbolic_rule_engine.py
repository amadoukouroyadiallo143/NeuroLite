import pytest
import json
import os
from neurolite.symbolic import SymbolicRuleEngine, MalformedFactError, MalformedRuleError
from collections import defaultdict

# Helper to create a temporary rules file
@pytest.fixture
def temp_rules_file(tmp_path):
    rules_content = {
        "facts": ["estParentDe(jean, paul)", "estOiseau(titi)", "peutVoler(titi)", "estOiseau(pingu)"],
        "rules": [
            {
                "premises": ["estParentDe(?x, ?y)", "estParentDe(?y, ?z)"],
                "conclusion": "estGrandParentDe(?x, ?z)"
            },
            {
                "premises": ["estOiseau(?x)", "NOT peutVoler(?x)"],
                "conclusion": "estManchot(?x)"
            }
        ]
    }
    file_path = tmp_path / "test_rules.json"
    with open(file_path, 'w') as f:
        json.dump(rules_content, f)
    return str(file_path)

@pytest.fixture
def empty_rules_file(tmp_path):
    file_path = tmp_path / "empty_rules.json"
    with open(file_path, 'w') as f:
        json.dump({"facts": [], "rules": []}, f)
    return str(file_path)

def test_add_fact():
    engine = SymbolicRuleEngine()
    engine.add_fact("estUn(chien, animal)")
    assert "estUn(chien, animal)" in engine.facts["estUn"]
    engine.add_fact("couleur(chien, brun)")
    assert "couleur(chien, brun)" in engine.facts["couleur"]
    assert len(engine.facts["estUn"]) == 1
    assert len(engine.facts["couleur"]) == 1

def test_add_fact_malformed():
    engine = SymbolicRuleEngine()
    with pytest.raises(MalformedFactError):
        engine.add_fact("NOT estUn(chat, animal)") # Facts cannot be negative
    with pytest.raises(MalformedFactError):
        engine.add_fact("faitMalForme") # Missing parentheses for args
    with pytest.raises(MalformedFactError):
        engine.add_fact("") # Empty fact
    with pytest.raises(MalformedFactError):
        engine.add_fact("  ") # Blank fact

def test_add_rule():
    engine = SymbolicRuleEngine()
    premises = ["estParentDe(?x, ?y)", "estParentDe(?y, ?z)"]
    conclusion = "estGrandParentDe(?x, ?z)"
    engine.add_rule(premises, conclusion)
    assert (premises, conclusion) in engine.rules

def test_add_rule_malformed():
    engine = SymbolicRuleEngine()
    with pytest.raises(MalformedRuleError):
        engine.add_rule([], "conclusionInvalide(?x)") # Empty premises
    with pytest.raises(MalformedRuleError):
        engine.add_rule(["premisseValide(?x)"], "NOT conclusionNegative(?x)") # Negative conclusion
    with pytest.raises(MalformedRuleError):
        engine.add_rule([""], "conclusion(?x)") # Empty premise
    with pytest.raises(MalformedRuleError):
        engine.add_rule(["premisseValide(?x)"], "") # Empty conclusion
    with pytest.raises(MalformedRuleError):
        engine.add_rule(["premisseMalFormee"], "conclusion(?x)")


def test_load_rules_from_file(temp_rules_file):
    engine = SymbolicRuleEngine(rules_file=temp_rules_file)
    assert "estParentDe(jean, paul)" in engine.facts["estParentDe"]
    assert "estOiseau(titi)" in engine.facts["estOiseau"]
    assert "peutVoler(titi)" in engine.facts["peutVoler"]
    assert "estOiseau(pingu)" in engine.facts["estOiseau"]
    
    expected_rule_1 = (["estParentDe(?x, ?y)", "estParentDe(?y, ?z)"], "estGrandParentDe(?x, ?z)")
    expected_rule_2 = (["estOiseau(?x)", "NOT peutVoler(?x)"], "estManchot(?x)")
    assert expected_rule_1 in engine.rules
    assert expected_rule_2 in engine.rules

def test_load_rules_file_not_found():
    with pytest.raises(FileNotFoundError):
        SymbolicRuleEngine(rules_file="non_existent_rules.json")

def test_load_rules_invalid_json(tmp_path):
    invalid_json_file = tmp_path / "invalid.json"
    with open(invalid_json_file, 'w') as f:
        f.write("{'facts': ['a(b)']") # Invalid JSON
    with pytest.raises(json.JSONDecodeError):
        SymbolicRuleEngine(rules_file=str(invalid_json_file))

def test_basic_inference_simple_rule(empty_rules_file):
    engine = SymbolicRuleEngine(rules_file=empty_rules_file) # Start with an empty engine
    engine.add_fact("A(1)")
    engine.add_fact("A(2)")
    engine.add_rule(["A(?x)"], "B(?x)")
    derived_facts = engine.infer()
    assert "A(1)" in derived_facts
    assert "A(2)" in derived_facts
    assert "B(1)" in derived_facts
    assert "B(2)" in derived_facts

def test_basic_inference_transitivity(empty_rules_file):
    engine = SymbolicRuleEngine(rules_file=empty_rules_file)
    engine.add_fact("R(a, b)")
    engine.add_fact("R(b, c)")
    engine.add_fact("R(c, d)") # For more checks
    engine.add_rule(["R(?x, ?y)", "R(?y, ?z)"], "R_trans(?x, ?z)")
    derived_facts = engine.infer()
    
    assert "R(a, b)" in derived_facts
    assert "R(b, c)" in derived_facts
    assert "R(c, d)" in derived_facts
    assert "R_trans(a, c)" in derived_facts
    assert "R_trans(b, d)" in derived_facts
    assert "R_trans(a, d)" not in derived_facts # Requires two steps, current infer does one pass effectively per iteration

    # Test multi-hop if infer handles iterations correctly
    # The current infer will find (a,c) and (b,d) in the first iter.
    # To get (a,d) from R_trans(a,c) and R(c,d) -> R_trans(a,d) would need R_trans to be a premise.
    # Or, if the rule was R(?x,?y), R_trans(?y,?z) -> R_trans(?x,?z)
    # For now, this tests single application of the rule.

def test_inference_with_negation(empty_rules_file):
    engine = SymbolicRuleEngine(rules_file=empty_rules_file)
    engine.add_fact("IsBird(tweety)")
    engine.add_fact("CanFly(tweety)")
    engine.add_fact("IsBird(pingu)")
    # Implicitly, CanFly(pingu) is not a fact
    engine.add_fact("IsBird(roadrunner)")
    engine.add_fact("CanFly(roadrunner)")
    engine.add_fact("IsPlane(boeing747)")
    engine.add_fact("CanFly(boeing747)")

    engine.add_rule(["IsBird(?x)", "NOT CanFly(?x)"], "IsPenguin(?x)")
    derived_facts = engine.infer()

    assert "IsBird(tweety)" in derived_facts
    assert "CanFly(tweety)" in derived_facts
    assert "IsBird(pingu)" in derived_facts
    assert "IsPenguin(pingu)" in derived_facts
    assert "IsPenguin(tweety)" not in derived_facts
    assert "IsPenguin(roadrunner)" not in derived_facts
    assert "IsPenguin(boeing747)" not in derived_facts

def test_inference_no_new_facts(empty_rules_file):
    engine = SymbolicRuleEngine(rules_file=empty_rules_file)
    engine.add_fact("A(1)")
    engine.add_rule(["B(?x)"], "C(?x)") # Rule that won't fire
    derived_facts = engine.infer()
    assert derived_facts == {"A(1)"}

def test_inference_predicate_without_args(empty_rules_file):
    engine = SymbolicRuleEngine(rules_file=empty_rules_file)
    engine.add_fact("pleut")
    engine.add_rule(["pleut"], "solEstMouillé")
    derived_facts = engine.infer()
    assert "pleut" in derived_facts
    assert "solEstMouillé" in derived_facts

    engine.add_fact("faitBeau")
    engine.add_rule(["faitBeau", "NOT pleut"], "allerAuParc")
    derived_facts_2 = engine.infer() # Re-infer after adding new rule/fact
    assert "faitBeau" in derived_facts_2
    assert "allerAuParc" in derived_facts_2


def test_parse_predicate_helper():
    # This is a private method, but crucial. Testing it directly can be useful.
    # If it's strictly private and not intended for direct test, these can be removed/adapted.
    engine = SymbolicRuleEngine()
    name, args, negated = engine._parse_predicate("pred(a,b)")
    assert name == "pred"
    assert args == ["a", "b"]
    assert not negated

    name, args, negated = engine._parse_predicate("NOT pred(a, ?x, c)")
    assert name == "pred"
    assert args == ["a", "?x", "c"]
    assert negated

    name, args, negated = engine._parse_predicate("simplePred")
    assert name == "simplePred"
    assert args == []
    assert not negated
    
    name, args, negated = engine._parse_predicate("NOT complex()")
    assert name == "complex"
    assert args == []
    assert negated

    with pytest.raises(MalformedFactError):
        engine._parse_predicate("pred(a,b") # Missing closing parenthesis
    with pytest.raises(MalformedFactError):
        engine._parse_predicate("pred a b)")
    with pytest.raises(MalformedFactError):
        engine._parse_predicate("NOT pred(a,b")

def test_malformed_rules_in_load(tmp_path):
    # Malformed facts in rules file
    rules_content_bad_fact = {
        "facts": ["estParentDe(jean, paul", "estOiseau(titi)"], # Malformed fact
        "rules": []
    }
    file_path_bad_fact = tmp_path / "bad_fact_rules.json"
    with open(file_path_bad_fact, 'w') as f:
        json.dump(rules_content_bad_fact, f)
    # This should be caught by add_fact via load_rules if _parse_predicate is used there effectively
    # The current SymbolicRuleEngine.load_rules calls add_fact, which calls _parse_predicate.
    # However, _parse_predicate is mostly for format validation, not content.
    # add_fact itself does not re-validate the full string with _parse_predicate if it's not malformed structurally.
    # Let's ensure load_rules properly validates.
    # The current `add_fact` uses `_parse_predicate` which should catch this.
    # Let's test the error propagation from load_rules
    with pytest.raises(MalformedFactError): # Assuming _parse_predicate in add_fact catches it.
         SymbolicRuleEngine(rules_file=file_path_bad_fact)


    # Malformed rules in rules file
    rules_content_bad_rule = {
        "facts": ["estParentDe(jean, paul)"],
        "rules": [
            {
                "premises": ["estParentDe(?x, ?y)", "NOT estParentDe(?y, ?z"], # Malformed premise
                "conclusion": "estGrandParentDe(?x, ?z)"
            }
        ]
    }
    file_path_bad_rule = tmp_path / "bad_rule_rules.json"
    with open(file_path_bad_rule, 'w') as f:
        json.dump(rules_content_bad_rule, f)
    # This should be caught by add_rule via load_rules.
    with pytest.raises(MalformedRuleError): # Or MalformedFactError if premise parsing fails first
        SymbolicRuleEngine(rules_file=file_path_bad_rule)

    rules_content_neg_conclusion = {
        "facts": [],
        "rules": [{"premises": ["A(?x)"], "conclusion": "NOT B(?x)"}]
    }
    file_path_neg_conclusion = tmp_path / "neg_conclusion_rules.json"
    with open(file_path_neg_conclusion, 'w') as f:
        json.dump(rules_content_neg_conclusion, f)
    with pytest.raises(MalformedRuleError):
        SymbolicRuleEngine(rules_file=file_path_neg_conclusion)

def test_rule_with_constants_in_premise(empty_rules_file):
    engine = SymbolicRuleEngine(rules_file=empty_rules_file)
    engine.add_fact("type(obj1, widget)")
    engine.add_fact("couleur(obj1, bleu)")
    engine.add_fact("type(obj2, gadget)")
    engine.add_fact("couleur(obj2, bleu)")
    
    engine.add_rule(["type(?x, widget)", "couleur(?x, bleu)"], "estWidgetBleu(?x)")
    derived = engine.infer()
    assert "estWidgetBleu(obj1)" in derived
    assert "estWidgetBleu(obj2)" not in derived

    engine.add_fact("type(obj3, widget)")
    engine.add_fact("couleur(obj3, rouge)")
    engine.add_rule(["type(?x, widget)", "NOT couleur(?x, bleu)"], "estWidgetNonBleu(?x)")
    derived2 = engine.infer()
    assert "estWidgetNonBleu(obj3)" in derived2
    assert "estWidgetNonBleu(obj1)" not in derived2

def test_empty_facts_or_rules(empty_rules_file):
    engine = SymbolicRuleEngine(rules_file=empty_rules_file)
    engine.add_fact("A(1)")
    derived = engine.infer()
    assert derived == {"A(1)"} # No rules, only initial facts

    engine = SymbolicRuleEngine(rules_file=empty_rules_file)
    engine.add_rule(["A(?x)"], "B(?x)")
    derived = engine.infer()
    assert derived == set() # No facts, rule doesn't fire
