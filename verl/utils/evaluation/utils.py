
def _count_behavior_emergence(data, tokenizer):
        """Analyzes response texts for cognitive behavior keywords with multi-count per utterance."""
        
        # 1. Define Behavior Taxonomy (Expandable)
        BEHAVIOR_CATEGORIES = {
            'Self-reflection': [
                'verify', 're-examine', 'double-check', 'why', 'root cause', 
                'validate', 'abduct', 'question', 'check', 'but', 'wait',
                'confirm', 'scrutinize', 'analyze', 'investigate', 'probe'
            ],
            'Hypotheticals': [
                'suppose', 'assume', 'if', 'hypothetically', 'what if',
                'doubt', 'presume', 'envision', 'try to', 'imagine',
                'assuming', 'speculate', 'pretend', 'postulate', 'consider', 
            ],
            'Branching Analysis': [
                'alternatively', 'on the other hand', 'another approach',
                'option', 'another', 'otherwise', 'instead', 'conversely',
                'plan B', 'fallback', 'choose', 'diverge'
            ],
            'Uncertainty Markers': [
                'possibly', 'uncertain', 'seems', 'likely', 'probable',
                'might', 'tentatively', 'conceivably', 'surmise', 'perhaps',
                'maybe', 'perchance', 'plausibly', 'potentially', 'presumably', 'guess'
            ],
            'Meta-cognition': [
                'reflect', 'contemplate', 'meditate', 'deep thought',
                'introspect', 'ponder', 'ruminate', 'deliberate',
                'cogitate', 'theorize', 'philosophize'
            ]
        }
        
        behavior_counts = {k: 0 for k in BEHAVIOR_CATEGORIES}
        behavior_counts.update({f"appear_{k}": 0 for k in BEHAVIOR_CATEGORIES})
        total_tokens = 0 
    
        # 2. Enhanced English Tokenization 
        import re 
        from collections import defaultdict 
        
        # 3. Case-insensitive Search Preparation 
        keyword_map = defaultdict(list)
        for category, terms in BEHAVIOR_CATEGORIES.items(): 
            keyword_map[category] = [term.lower() for term in terms]

        # 4. Processing Pipeline 
        for i in range(len(data)):
            data_item = data[i]
            prompt_length = data_item.batch['prompts'].shape[-1] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum() 
            response_ids = data_item.batch['responses'][:valid_response_length] 
            response_text = tokenizer.decode(response_ids).lower()
            
            # 计算总词数（用于后续归一化）
            words = re.findall(r'\b\w+\b', response_text)
            total_tokens += len(words)
            
            # 检查每个类别的所有关键词/短语
            for category, terms in keyword_map.items():
                match = 0
                for term in terms:
                    # 使用字符串的count方法来计算短语出现的次数
                    count = response_text.count(term)
                    if count > 0:
                        behavior_counts[category] += count
                        match = 1
                behavior_counts[f"appear_{category}"] += match
    
        # 5. Normalization & Formatting 
        behavior_ratios = {}
        if total_tokens > 0:
            for category, count in behavior_counts.items(): 
                if category.startswith("appear"):
                    behavior_ratios[category] = round(count / len(data), 4)
                else:
                    behavior_ratios[category] = round(count / total_tokens, 4)
        
        return behavior_ratios 